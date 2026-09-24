"""Shared fixtures: a fast simulated server session on synthetic prices."""

from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone

import pytest

from aifx import engine
from aifx.data import SyntheticMarket
from aifx.news import analyze_lexicon
from aifx.store import stable_id
from aifx.timeutil import iso

UTC = timezone.utc
START = datetime(2026, 9, 22, 0, 10, tzinfo=UTC)  # a Tuesday: Monday's daily bar has just closed


def _shrink(mp):
    """Shrink walk-forward priors so simulated sessions run in seconds."""
    mp.setitem(engine.TIMEFRAMES, "15m", engine.Timeframe("15m", "15分足", "分", (1, 4, 16), 16, 8, 4, 800, 100.0,
                                                          "15m", 15))
    mp.setitem(engine.TIMEFRAMES, "1h", engine.Timeframe("1h", "1時間足", "時間", (1, 4, 24), 24, 8, 6, 800, 50.0, "1h", 60))
    mp.setitem(engine.TIMEFRAMES, "1d", engine.Timeframe("1d", "日足", "営業日", (1, 5, 10, 20), 20, 8, 5, 500, 20.0,
                                                         "1h", 0))


@pytest.fixture(autouse=True)
def small_timeframes(monkeypatch):
    _shrink(monkeypatch)


_MARKET = SyntheticMarket(seed=5, start=datetime(2024, 1, 1, tzinfo=UTC))


def market():
    return _MARKET


def fake_news(at):
    rows = [
        ("Dollar surges as Fed turns hawkish", "en", 30),
        ("円急落、日銀が緩和維持", "ja", 50),
        ("Euro slides after weak German data misses expectations", "en", 70),
    ]
    items = []
    for title, lang, minutes in rows:
        t = f"{title} {at:%d%H}"
        items.append({"id": stable_id(t), "src": "gn-en-fx", "publisher": "Test", "title": t,
                      "link": "https://example.com/" + stable_id(t), "published_at": iso(at - timedelta(minutes=minutes)),
                      "lang": lang, "an": analyze_lexicon(title, lang)})
    return items, {"gn-en-fx": {"fetched": len(items), "relevant": len(items), "error": None}}


def fake_calendar():
    return [
        {"id": "ev-cpi", "cur": "USD", "title": "CPI m/m", "time": "2026-09-22T12:30:00Z", "impact": "High",
         "forecast": "0.3%", "previous": "0.2%"},
        {"id": "ev-boj", "cur": "JPY", "title": "BOJ Policy Rate", "time": "2026-09-23T03:00:00Z", "impact": "High",
         "forecast": "", "previous": ""},
    ], None


def fake_rates(at):
    """A rates item like rates.collect_rates makes: USD 4 %, JPY 0.5 %, EUR 2 %, GBP 4.5 %, AUD 4 %."""
    from aifx.history import RATE_SERIES
    level = {"USD": 4.0, "JPY": 0.5, "EUR": 2.0, "GBP": 4.5, "AUD": 4.0}
    series = {}
    for cur, items in RATE_SERIES.items():
        for sid, freq in items:
            step = timedelta(days=1 if freq == "d" else 30)
            n = 460 if freq == "d" else 30
            series[sid] = [[(at - step * k).strftime("%Y-%m-%d"), level[cur]] for k in range(n, 0, -1)]
    day = at.strftime("%Y-%m-%d")
    return {"id": f"rates-{day}", "date": day, "series": series}, None


def run_session(root, cycles=12, step_minutes=30, start=START, **kw):
    from aifx.pipeline import run_cycle

    mk = kw.pop("market", None) or market()
    kw.setdefault("backtest_budget", 0)       # the rolling backtest has its own tests
    reports = []
    for i in range(cycles):
        at = start + timedelta(minutes=step_minutes * i)
        kw.setdefault("rates_fetch", fake_rates)
        reports.append(run_cycle(root, now=at, market=mk, news_fetch=fake_news, calendar_fetch=fake_calendar,
                                 pairs=["USDJPY", "EURUSD"], log=None, **kw))
    return reports


@pytest.fixture(scope="session")
def base_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("base") / "state"
    with pytest.MonkeyPatch.context() as mp:
        _shrink(mp)
        reports = run_session(root, cycles=14)
    return root, reports


@pytest.fixture
def session(base_session, tmp_path):
    """A private copy of one simulated 7-hour session (tests may tamper with it)."""
    src, reports = base_session
    dst = tmp_path / "state"
    shutil.copytree(src, dst)
    return dst, reports
