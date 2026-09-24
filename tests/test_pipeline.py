"""End-to-end: a simulated server session, and attempts to cheat that must be caught."""

import json
import threading
import urllib.request
from datetime import timedelta

import pytest

import aifx.pipeline as pl
from aifx.audit import audit, verify
from aifx.ledger import Ledger
from aifx.timeutil import iso, parse_iso, utcnow

from .conftest import market, run_session


def kinds(rep):
    return {p["kind"] for p in rep["problems"]}


def test_session_issues_scores_learns_and_verifies(session):
    root, reports = session
    assert all(r.verify["ok"] for r in reports), reports[-1].verify["problems"][:3]
    ledger = Ledger(root).load()
    preds = ledger.of_type("prediction")
    origins = [(p["pair"], p["tf"], p["origin"]) for p in preds]
    assert len(origins) == len(set(origins))
    hourly = [p for p in preds if p["tf"] == "1h"]
    daily = [p for p in preds if p["tf"] == "1d"]
    assert len(hourly) >= 2 * 6 and len(daily) == 2
    outcomes = ledger.of_type("outcome")
    assert outcomes, "hourly forecasts should have matured during the session"
    by_seq = {p["seq"]: p for p in preds}
    for o in outcomes:
        for seq, h, actual, bar_end in o["items"]:
            f = next(x for x in by_seq[seq]["fc"] if x["h"] == h)
            assert o["at"] >= f["t"] and bar_end <= f["t"] and actual is not None
    rep = audit(root, sample=6)
    assert rep["ok"] and rep["checked"] >= 6


def test_first_forecasts_cannot_use_news_fetched_in_the_same_cycle(session):
    root, _ = session
    preds = Ledger(root).load().of_type("prediction")
    first_at = preds[0]["at"]
    assert all(p["news"]["n"] == 0 and p["news"]["x"] == 0 for p in preds if p["at"] == first_at)
    assert any(p["news"]["n"] > 0 for p in preds), "later forecasts should see headlines stored earlier"


def test_learning_state_uses_only_earlier_outcomes(session):
    root, _ = session
    ledger = Ledger(root).load()
    for p in ledger.of_type("prediction"):
        assert p["learn"] < p["seq"] and p["prior"] < p["seq"]


def test_peeking_at_future_prices_is_caught_by_the_audit(tmp_path, monkeypatch):
    mk = market()
    real = pl.make_prediction

    def peeking(tf, pair, bars, hourly, origin, *args, **kwargs):
        if tf.key == "1h":
            future, _ = mk.hourly(pair, origin + timedelta(hours=6))
            bars = future  # includes bars that closed after the origin
        return real(tf, pair, bars, hourly, origin, *args, **kwargs)

    monkeypatch.setattr(pl, "make_prediction", peeking)
    root = tmp_path / "state"
    run_session(root, cycles=6, market=mk)
    rep = audit(root, sample=6)
    assert not rep["ok"]
    assert any(not r["ok"] for r in rep["results"] if r["tf"] == "1h")


def test_skipping_bad_outcomes_is_caught(tmp_path, monkeypatch):
    real = pl.score_due
    monkeypatch.setattr(pl, "score_due", lambda state, at: [it for it in real(state, at) if it[0] % 2 == 0])
    reports = run_session(tmp_path / "state", cycles=8)
    final = reports[-1].verify
    assert not final["ok"]
    assert any("never scored" in p["msg"] for p in final["problems"])


def test_scoring_before_the_target_is_caught(tmp_path, monkeypatch):
    real = pl.score_due

    def early(state, at):
        items = real(state, at)
        done = {(i[0], i[1]) for i in items} | {(s, h) for r in state.ledger.of_type("outcome") for s, h, *_ in r["items"]}
        for p in state.ledger.of_type("prediction"):
            hourly = state.prices.load(p["pair"], "1h")
            for f in p["fc"]:
                if (p["seq"], f["h"]) not in done and parse_iso(f["t"]) > at:
                    items.append([p["seq"], f["h"], float(hourly["close"].iloc[-1]), iso(hourly.index[-1].to_pydatetime() + timedelta(hours=1))])
                    done.add((p["seq"], f["h"]))
        return items

    monkeypatch.setattr(pl, "score_due", early)
    reports = run_session(tmp_path / "state", cycles=4)
    final = reports[-1].verify
    assert not final["ok"]
    assert any("before its target" in p["msg"] for p in final["problems"])


def test_backdated_news_is_caught(tmp_path, monkeypatch):
    from aifx.store import NewsStore
    real = NewsStore.append

    def backdated(self, items, at):
        return real(self, items, at - timedelta(hours=2))

    monkeypatch.setattr(NewsStore, "append", backdated)
    reports = run_session(tmp_path / "state", cycles=3)
    final = reports[-1].verify
    assert not final["ok"] and "time" in kinds(final)


def test_forged_prediction_with_convenient_targets_is_caught(session):
    root, _ = session
    ledger = Ledger(root).load()
    p = dict(ledger.of_type("prediction")[-1])
    for key in ("seq", "prev", "hash", "at"):
        p.pop(key)
    p["fc"] = [dict(f, t=iso(parse_iso(f["t"]) + timedelta(hours=1))) for f in p["fc"]]
    ledger.append(p, parse_iso(ledger.records[-1]["at"]) + timedelta(minutes=1))
    rep = verify(root)
    msgs = " ".join(x["msg"] for x in rep["problems"])
    assert "duplicate" in msgs and "targets differ" in msgs


@pytest.mark.parametrize("target", ["prices", "news", "ledger"])
def test_editing_stored_files_is_caught(session, target):
    root, _ = session
    if target == "ledger":
        path = next((root / "ledger").glob("*.jsonl"))
        lines = path.read_text().splitlines()
        rec = json.loads(lines[-1]); rec["at"] = "2026-01-01T00:00:00Z"; lines[-1] = json.dumps(rec)
    else:
        path = next((root / target).glob("*"))
        lines = path.read_text().splitlines()
        lines[-1] = lines[-1].replace("0", "1", 1)
    path.write_text("\n".join(lines) + "\n")
    rep = verify(root)
    assert not rep["ok"] and kinds(rep) & {"chain", "data"}


def test_api_documents_are_strict_json(session, tmp_path):
    from aifx.api import build_api, write_api
    from aifx.site import write_site
    root, _ = session
    docs = build_api(root)
    for name, obj in docs.items():
        json.dumps(obj, allow_nan=False)
    site = tmp_path / "site"
    write_site(site)
    write_api(docs, site)
    meta = json.loads((site / "api" / "meta.json").read_text())
    assert meta["ledger"]["ok"] and meta["pairs"] and meta["pairs"][0]["outlook"]["1h"]
    pair = json.loads((site / "api" / "pair" / "USDJPY.json").read_text())
    assert pair["tf"]["1h"]["path"]["steps"] and pair["tf"]["1h"]["prediction"]["horizons"]
    h = pair["tf"]["1h"]["prediction"]["horizons"][-1]
    assert h["lo95"] < h["lo80"] < h["lo50"] < h["hi50"] < h["hi80"] < h["hi95"]
    assert h["dist"]["levels"] and h["dist"]["curve"]
    market = json.loads((site / "api" / "market.json").read_text())
    assert market["pairs"] and all(x["volatility"] is None or 0 <= x["volatility"]["percentile"] <= 1
                                   for x in market["pairs"])
    if len(market["pairs"]) >= 3:        # relative strength needs enough pairs to be identified
        assert set(market["strength"]["24h"]) == {"USD", "JPY", "EUR", "GBP", "AUD"}
    html = (site / "index.html").read_text()
    assert html.startswith("<!doctype html>") and "meta.json" in html and "<title>AIFX 為替予測</title>" in html


def test_server_status_and_rate_limited_refresh(tmp_path):
    import functools
    import http.server

    from aifx.server import Handler, Scheduler
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("ok")
    sched = Scheduler(tmp_path / "state", site, 5, {})
    Handler.scheduler = sched
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(site)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        st = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert st["running"] is False and st["interval_min"] == 5
        req = urllib.request.Request(base + "/api/refresh", method="POST")
        assert urllib.request.urlopen(req).status == 202
        sched.last_started = utcnow()
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(urllib.request.Request(base + "/api/refresh", method="POST"))
        assert err.value.code == 429
    finally:
        httpd.shutdown()


def test_api_is_rebuilt_from_the_ledger_alone(session):
    import shutil

    from aifx.api import build_api
    root, _ = session
    shutil.rmtree(root / "cache")
    docs = build_api(root)
    blk = docs["pair/EURUSD.json"]["tf"]["1h"]
    assert blk["path"]["steps"] and blk["prediction"]["horizons"]
    assert docs["models.json"]["learning"]["1h"]["1"]["n"] >= 1
    assert (root / ".gitignore").read_text().strip() == "cache/"


def test_documented_verification_commands_work_on_a_fresh_checkout(session, monkeypatch):
    """What a third party runs after cloning the ledger branch (no cache directory)."""
    import shutil

    from aifx import cli
    from aifx.data import SyntheticMarket
    root, _ = session
    shutil.rmtree(root / "cache")
    assert cli.main(["verify", "--state", str(root)]) == 0
    head = Ledger(root).load().head
    assert cli.main(["verify", "--state", str(root), "--expect-head", f"{head[0]}:{head[1]}"]) == 0
    assert cli.main(["verify", "--state", str(root), "--expect-head", f"{head[0]}:{'0' * 64}"]) == 2
    monkeypatch.setattr("aifx.data.YahooMarket", lambda: market())
    assert cli.main(["audit", "--state", str(root), "--sample", "3", "--external"]) == 0
    assert (root / "cache" / "external.json").exists()


def test_fifteen_minute_forecasts_are_issued_scored_on_15m_bars_and_audited(session):
    root, _ = session
    ledger = Ledger(root).load()
    preds = [p for p in ledger.of_type("prediction") if p["tf"] == "15m"]
    assert len(preds) >= 2 * 10
    for p in preds:
        assert parse_iso(p["origin"]).minute % 15 == 0 and p["p0_bar"] == p["origin"]
        assert [f["h"] for f in p["fc"]] == [1, 4, 16]
    by_seq = {p["seq"]: p for p in preds}
    items = [it for o in ledger.of_type("outcome") for it in o["items"] if it[0] in by_seq]
    assert items, "15-minute forecasts should have been scored"
    for seq, h, actual, bar_end in items:
        f = next(x for x in by_seq[seq]["fc"] if x["h"] == h)
        assert bar_end == f["t"] and actual is not None     # the 15-minute bar closing at the target
    rep = audit(root, sample=12)
    assert rep["ok"] and any(r["tf"] == "15m" for r in rep["results"])


def test_peeking_at_future_15m_bars_is_caught(tmp_path, monkeypatch):
    mk = market()
    real = pl.make_prediction

    def peeking(tf, pair, bars, ref, origin, *args, **kwargs):
        if tf.key == "15m":
            bars, _ = mk.intraday(pair, origin + timedelta(hours=2), 15, "5d")
        return real(tf, pair, bars, ref, origin, *args, **kwargs)

    monkeypatch.setattr(pl, "make_prediction", peeking)
    root = tmp_path / "state"
    run_session(root, cycles=5, market=mk)
    rep = audit(root, sample=8)
    assert any(not r["ok"] for r in rep["results"] if r["tf"] == "15m")


def test_scoring_a_15m_forecast_with_the_wrong_bar_is_caught(tmp_path, monkeypatch):
    real = pl.score_due

    def hourly_instead(state, at):
        items = real(state, at)
        preds = {p["seq"]: p for p in state.ledger.of_type("prediction")}
        out = []
        for seq, h, actual, bar_end in items:
            if actual is not None and preds[seq]["tf"] == "15m":
                hourly = state.prices.load(preds[seq]["pair"], "1h")
                actual = float(hourly["close"].iloc[-1]) * 1.001
            out.append([seq, h, actual, bar_end])
        return out

    monkeypatch.setattr(pl, "score_due", hourly_instead)
    reports = run_session(tmp_path / "state", cycles=5)
    final = reports[-1].verify
    assert not final["ok"] and any("not the committed bar" in p["msg"] for p in final["problems"])


def test_fifteen_minute_targets_skip_the_weekend():
    from datetime import datetime, timezone
    from aifx.engine import TIMEFRAMES, target_times
    origin = datetime(2026, 9, 25, 20, 45, tzinfo=timezone.utc)   # Friday 16:45 New York
    t = target_times(TIMEFRAMES["15m"], origin)
    assert t[0] == datetime(2026, 9, 25, 21, 0, tzinfo=timezone.utc)                 # the last bar before the close
    assert t[1] == datetime(2026, 9, 27, 21, 45, tzinfo=timezone.utc)                # Sunday after the open
    assert TIMEFRAMES["15m"].horizon_label(4) == "1時間後" and TIMEFRAMES["15m"].horizon_label(1) == "15分後"


def test_no_15m_forecast_when_its_first_target_has_passed(tmp_path):
    from datetime import datetime, timezone
    # 12:16: the 12:00-12:15 bar is closed but not settled yet, so the newest stored bar
    # ends at 12:00 and a "15 minutes ahead" forecast would target 12:15, already past.
    at = datetime(2026, 9, 22, 12, 16, tzinfo=timezone.utc)
    rep = run_session(tmp_path / "state", cycles=1, start=at)[0]
    preds = Ledger(tmp_path / "state").load().of_type("prediction")
    assert rep.verify["ok"] and not [p for p in preds if p["tf"] == "15m"]
