"""One server cycle: collect -> commit inputs -> score -> learn -> predict -> verify.

Order matters. Raw inputs are committed (``batch`` record) before anything
uses them; outcomes are recorded before the learning state is derived; new
predictions come last and may only use data committed earlier in the chain.
"""

from __future__ import annotations

import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from . import news as newsmod
from .data import PAIRS, SETTLE, Pair, YahooMarket
from .engine import TIMEFRAMES, Timeframe, backtest_prior
from .forecaster import MAX_ORIGIN_AGE, make_prediction, model_version, origin_of
from .learning import learn, samples_from_ledger
from .ledger import DataFiles, Ledger
from .store import CalendarStore, NewsStore, PriceStore
from .timeutil import add_business_days, iso, london_day_end, parse_iso, utcnow


@dataclass
class State:
    root: Path
    ledger: Ledger
    files: DataFiles
    prices: PriceStore
    news: NewsStore
    calendar: CalendarStore

    @classmethod
    def open(cls, root: Path | str) -> "State":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        ignore = root / ".gitignore"
        if not ignore.exists():  # derived caches are rebuilt from the ledger, never committed
            ignore.write_text("cache/\n", encoding="utf-8")
        ledger = Ledger(root).load()
        files = DataFiles(root, ledger)
        return cls(root, ledger, files, PriceStore(files), NewsStore(files), CalendarStore(files))

    def cache_path(self, name: str) -> Path:
        p = self.root / "cache" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def read_cache(self, name: str, default=None):
        p = self.root / "cache" / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return default

    def write_cache(self, name: str, obj) -> None:
        self.cache_path(name).write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


@dataclass
class CycleReport:
    at: str
    appended: dict = field(default_factory=dict)
    predictions: int = 0
    outcomes: int = 0
    priors: list = field(default_factory=list)
    news: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    quotes: dict = field(default_factory=dict)
    verify: dict = field(default_factory=dict)


def _latest(records: list[dict], **match) -> dict | None:
    for rec in reversed(records):
        if all(rec.get(k) == v for k, v in match.items()):
            return rec
    return None


def score_due(state: State, at: datetime) -> list[list]:
    """Outcome items for every forecast horizon whose target time has passed and
    for which the hourly data now extends past the target."""
    scored = {(s, h) for rec in state.ledger.of_type("outcome") for s, h, *_ in rec["items"]}
    items = []
    hourly_cache: dict[str, pd.DataFrame] = {}
    for p in state.ledger.of_type("prediction"):
        pending = [f for f in p["fc"] if (p["seq"], f["h"]) not in scored]
        if not pending:
            continue
        if p["pair"] not in hourly_cache:
            hourly_cache[p["pair"]] = state.prices.load(p["pair"], "1h")
        hourly = hourly_cache[p["pair"]]
        if not len(hourly):
            continue
        ends = hourly.index + pd.Timedelta(hours=1)
        last_end = ends[-1].to_pydatetime()
        origin = parse_iso(p["origin"])
        for f in pending:
            t = parse_iso(f["t"])
            if t > at or last_end < t:
                continue
            idx = int(ends.searchsorted(pd.Timestamp(t), side="right")) - 1
            if idx >= 0 and ends[idx].to_pydatetime() > origin:
                items.append([p["seq"], f["h"], float(hourly["close"].iloc[idx]), iso(ends[idx].to_pydatetime())])
            else:
                items.append([p["seq"], f["h"], None, None])  # no data between origin and target: void
    return items


def _need_prior(state: State, tf: Timeframe, anchor: datetime, version: str) -> bool:
    rec = _latest(state.ledger.records, type="prior", tf=tf.key)
    # recomputed daily, and at once when the forecasting code changes
    return rec is None or rec["cutoff"] < iso(anchor) or rec.get("v", "") != version


def run_cycle(root: Path | str, now: datetime | None = None, market=None, collect_news=True,
              news_fetch=None, calendar_fetch=None, pairs: list[str] | None = None,
              timeframes=None, log=print) -> CycleReport:
    """Run one cycle against the state directory ``root``."""
    state = State.open(root)
    at = (now or utcnow()).replace(microsecond=0)
    if state.ledger.records and iso(at) <= state.ledger.records[-1]["at"]:
        at = parse_iso(state.ledger.records[-1]["at"]) + timedelta(seconds=1)
    report = CycleReport(at=iso(at))
    market = market or YahooMarket()
    pair_objs: list[Pair] = [PAIRS[c] for c in (pairs or list(PAIRS))]
    tfs: list[Timeframe] = [TIMEFRAMES[k] for k in (timeframes or list(TIMEFRAMES))]
    version = model_version()

    # 1. market data ------------------------------------------------------
    def fetch(pair: Pair):
        out = {"pair": pair.code, "hourly": None, "daily": None, "live": None, "errors": []}
        have_h = state.prices.load(pair.code, "1h")
        # Once history is stored, only the most recent bars need downloading.
        gap = (at - have_h.index[-1].to_pydatetime()) if len(have_h) else None
        range_ = "5d" if gap is not None and gap < timedelta(days=4) else "1mo" if gap is not None and gap < timedelta(days=25) else "1y"
        try:
            out["hourly"], out["live"] = market.hourly(pair, at, range_)
        except Exception as exc:
            out["errors"].append(f"{pair.code} 1h: {exc}")
        if any(tf.key == "1d" for tf in tfs):
            have = state.prices.load(pair.code, "1d")
            # Only ask for daily bars once the next business day could have completed.
            stale = not len(have) or london_day_end(add_business_days(have.index[-1].date(), 1)) + SETTLE <= at
            if stale:
                try:
                    out["daily"], _src = market.daily(pair, at)
                except Exception as exc:
                    out["errors"].append(f"{pair.code} 1d: {exc}")
        return out

    with ThreadPoolExecutor(max_workers=min(8, len(pair_objs))) as ex:
        fetched = list(ex.map(fetch, pair_objs))
    for f in fetched:
        report.errors.extend(f["errors"])
        if f["hourly"] is not None:
            report.appended[f"{f['pair']}_1h"] = state.prices.append_new(f["pair"], "1h", f["hourly"])
        if f["daily"] is not None:
            report.appended[f"{f['pair']}_1d"] = state.prices.append_new(f["pair"], "1d", f["daily"])
        if f["live"] and f["live"].get("price"):
            report.quotes[f["pair"]] = f["live"]

    # 2. news and calendar ------------------------------------------------
    if collect_news:
        items, src_report = (news_fetch or newsmod.collect_news)(at)
        known = {it["id"] for it in state.news.load(since=at - timedelta(days=62))}
        fresh = [it for it in items if it["id"] not in known]
        stored = state.news.append(fresh, at)
        events, cal_err = (calendar_fetch or newsmod.collect_calendar)()
        new_events = state.calendar.append(events, at)
        report.news = {
            "fetched": len(items), "new": len(stored), "events_new": len(new_events), "sources": src_report,
            "analyzer": newsmod.ANALYZER, "calendar_error": cal_err,
        }
        for err in [e["error"] for e in src_report.values() if e.get("error")] + [cal_err]:
            if err:
                report.errors.append(err)

    # 3. commit inputs -----------------------------------------------------
    state.files.commit(at, {"cycle": {"version": version, "git": os.environ.get("GITHUB_SHA", "")[:12],
                                      "run": os.environ.get("GITHUB_RUN_ID", "")}})

    # 4. score matured forecasts -------------------------------------------
    items = score_due(state, at)
    if items:
        state.ledger.append({"type": "outcome", "items": items}, at)
    report.outcomes = len(items)

    # 5. walk-forward priors (once per UTC day) ----------------------------
    anchor = at.replace(hour=0, minute=0, second=0)
    for tf in tfs:
        if _need_prior(state, tf, anchor, version):
            series = {p.code: state.prices.load(p.code, tf.key) for p in pair_objs}
            hourly = {p.code: state.prices.load(p.code, "1h") for p in pair_objs} if tf.key == "1d" else None
            try:
                prior, _detail = backtest_prior(tf, series, anchor, hourly, version)
            except Exception as exc:
                report.errors.append(f"prior {tf.key}: {exc}")
                continue
            if prior["h"]:
                state.ledger.append({"type": "prior", **prior}, at)
                report.priors.append(tf.key)

    # 6. learning state + 7. predictions ------------------------------------
    predictions = {p["seq"]: p for p in state.ledger.of_type("prediction")}
    outcome_recs = state.ledger.of_type("outcome")
    learn_seq = outcome_recs[-1]["seq"] if outcome_recs else 0
    news_items = state.news.load(since=at - timedelta(days=4))
    events = state.calendar.load(since=at - timedelta(days=40))
    latest = state.read_cache("latest.json", {})
    for tf in tfs:
        prior_rec = _latest(state.ledger.records, type="prior", tf=tf.key)
        if prior_rec is None:
            continue
        samples = samples_from_ledger(predictions, outcome_recs, tf.key)
        st = learn(prior_rec, samples, tf.horizons, tf.half_life)
        issued = {(p["pair"], p["origin"]) for p in predictions.values() if p["tf"] == tf.key}
        for pair in pair_objs:
            bars = state.prices.load(pair.code, tf.key)
            hourly = state.prices.load(pair.code, "1h")
            if len(bars) < 400 or not len(hourly):
                continue
            origin = origin_of(tf, bars)
            if (pair.code, iso(origin)) in issued or at - origin > MAX_ORIGIN_AGE[tf.key] or origin > at:
                continue
            try:
                rec, chart = make_prediction(tf, pair, bars, hourly, origin, news_items, events,
                                             prior_rec["seq"], learn_seq, st, version)
            except Exception as exc:
                report.errors.append(f"{pair.code} {tf.key} forecast: {exc}")
                traceback.print_exc()
                continue
            full = state.ledger.append(rec, at)
            chart["seq"] = full["seq"]
            latest.setdefault(pair.code, {})[tf.key] = chart
            report.predictions += 1
    state.write_cache("latest.json", latest)

    # 8. verify everything written so far -----------------------------------
    from .audit import verify
    report.verify = verify(state.root)
    status = {
        "at": report.at, "quotes": report.quotes, "errors": report.errors[:50], "news": report.news,
        "appended": report.appended, "predictions": report.predictions, "outcomes": report.outcomes,
        "priors": report.priors, "verify_ok": report.verify["ok"],
    }
    state.write_cache("status.json", status)
    if log:
        log(f"cycle {report.at}: +{report.predictions} predictions, +{report.outcomes} outcomes, "
            f"news +{report.news.get('new', 0)}, verify {'OK' if report.verify['ok'] else 'FAILED'}")
    return report
