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
