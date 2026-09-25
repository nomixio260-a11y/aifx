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

from . import backtest
from . import rates as ratesmod
from . import news as newsmod
from .data import PAIRS, SETTLE, Pair, YahooMarket
from .engine import TIMEFRAMES, Timeframe, backtest_prior, target_times
from .forecaster import MAX_ORIGIN_AGE, make_prediction, model_version, origin_of
from .learning import learn, samples_from_ledger
from .ledger import DataFiles, Ledger
from .store import CalendarStore, NewsStore, PriceStore, RatesStore
from .timeutil import add_business_days, iso, london_day_end, parse_iso, utcnow


@dataclass
class State:
    root: Path
    ledger: Ledger
    files: DataFiles
    prices: PriceStore
    news: NewsStore
    calendar: CalendarStore
    rates: RatesStore

    @classmethod
    def open(cls, root: Path | str) -> "State":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        ignore = root / ".gitignore"
        if not ignore.exists():  # derived caches are rebuilt from the ledger, never committed
            ignore.write_text("cache/\n", encoding="utf-8")
        ledger = Ledger(root).load()
        files = DataFiles(root, ledger)
        return cls(root, ledger, files, PriceStore(files), NewsStore(files), CalendarStore(files), RatesStore(files))

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
    backtest: dict = field(default_factory=dict)


def _latest(records: list[dict], **match) -> dict | None:
    for rec in reversed(records):
        if all(rec.get(k) == v for k, v in match.items()):
            return rec
    return None


def score_due(state: State, at: datetime) -> list[list]:
    """Outcome items for every forecast horizon whose target time has passed and
    for which the reference bars of its timeframe now extend past the target."""
    scored = {(s, h) for rec in state.ledger.of_type("outcome") for s, h, *_ in rec["items"]}
    items = []
    ref_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for p in state.ledger.of_type("prediction"):
        pending = [f for f in p["fc"] if (p["seq"], f["h"]) not in scored]
        if not pending:
            continue
        ref_tf = TIMEFRAMES[p["tf"]].ref
        key = (p["pair"], ref_tf)
        if key not in ref_cache:
            ref_cache[key] = state.prices.load(p["pair"], ref_tf)
        ref = ref_cache[key]
        if not len(ref):
            continue
        ends = ref.index + pd.Timedelta(minutes=TIMEFRAMES[ref_tf].minutes)
        last_end = ends[-1].to_pydatetime()
        origin = parse_iso(p["origin"])
        for f in pending:
            t = parse_iso(f["t"])
            if t > at or last_end < t:
                continue
            idx = int(ends.searchsorted(pd.Timestamp(t), side="right")) - 1
            if idx >= 0 and ends[idx].to_pydatetime() > origin:
                items.append([p["seq"], f["h"], float(ref["close"].iloc[idx]), iso(ends[idx].to_pydatetime())])
            else:
                items.append([p["seq"], f["h"], None, None])  # no data between origin and target: void
    return items


def _need_prior(state: State, tf: Timeframe, anchor: datetime, version: str) -> bool:
    rec = _latest(state.ledger.records, type="prior", tf=tf.key)
    # recomputed daily, and at once when the forecasting code changes
    return rec is None or rec["cutoff"] < iso(anchor) or rec.get("v", "") != version


def run_cycle(root: Path | str, now: datetime | None = None, market=None, collect_news=True,
              news_fetch=None, calendar_fetch=None, rates_fetch=None, pairs: list[str] | None = None,
              timeframes=None, log=print, backtest_budget: int | None = None) -> CycleReport:
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
        out = {"pair": pair.code, "hourly": None, "daily": None, "live": None, "intraday": {}, "errors": []}
        have_h = state.prices.load(pair.code, "1h")
        # Once history is stored, only the most recent bars need downloading.
        gap = (at - have_h.index[-1].to_pydatetime()) if len(have_h) else None
        range_ = "5d" if gap is not None and gap < timedelta(days=4) else "1mo" if gap is not None and gap < timedelta(days=25) else "1y"
        try:
            out["hourly"], out["live"] = market.hourly(pair, at, range_)
        except Exception as exc:
            out["errors"].append(f"{pair.code} 1h: {exc}")
        for tf in tfs:
            if tf.minutes and tf.key != "1h":
                have = state.prices.load(pair.code, tf.key)
                gap = (at - have.index[-1].to_pydatetime()) if len(have) else None
                # Yahoo keeps about 60 days of 15-minute bars.
                rng = "5d" if gap is not None and gap < timedelta(days=4) else "1mo" if gap is not None and gap < timedelta(days=25) else "60d"
                try:
                    out["intraday"][tf.key], live = market.intraday(pair, at, tf.minutes, rng)
                    out["live"] = out["live"] or live
                except Exception as exc:
                    out["errors"].append(f"{pair.code} {tf.key}: {exc}")
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
        for key, bars in f["intraday"].items():
            report.appended[f"{f['pair']}_{key}"] = state.prices.append_new(f["pair"], key, bars)
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
        # short rates for the carry rules: once a day
        if not any(it["date"] == at.strftime("%Y-%m-%d") for it in state.rates.load(since=at - timedelta(days=2))):
            item, rate_err = (rates_fetch or ratesmod.collect_rates)(at)
            if item is not None:
                state.rates.append([item], at)
            if rate_err:
                report.errors.append(rate_err)

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
    rate_items = state.rates.load(since=at - timedelta(days=40))
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
            ref = state.prices.load(pair.code, tf.ref)
            if len(bars) < 400 or not len(ref):
                continue
            origin = origin_of(tf, bars)
            if (pair.code, iso(origin)) in issued or at - origin > MAX_ORIGIN_AGE[tf.key] or origin > at:
                continue
            if target_times(tf, origin, tf.horizons[:1])[0] <= at:
                continue    # the next bar has already closed: a forecast for it would not be ahead of time
            hourly = ref if tf.ref == "1h" else state.prices.load(pair.code, "1h") if tf.minutes else None
            try:
                rec, chart = make_prediction(tf, pair, bars, ref, origin, news_items, events,
                                             prior_rec["seq"], learn_seq, st, version, rate_items, hourly)
            except Exception as exc:
                report.errors.append(f"{pair.code} {tf.key} forecast: {exc}")
                traceback.print_exc()
                continue
            full = state.ledger.append(rec, at)
            chart["seq"] = full["seq"]
            latest.setdefault(pair.code, {})[tf.key] = chart
            report.predictions += 1
    state.write_cache("latest.json", latest)

    # 8. rolling backtest on the stored history (derived; never part of the ledger)
    budget = backtest.BUDGET if backtest_budget is None else backtest_budget
    if budget:
        try:
            report.backtest = backtest.update(state, tfs, [p.code for p in pair_objs], at, budget=budget)
        except Exception as exc:
            report.errors.append(f"backtest: {exc}")
            traceback.print_exc()

    # 9. verify everything written so far -----------------------------------
    from .audit import verify
    report.verify = verify(state.root)
    status = {
        "at": report.at, "quotes": report.quotes, "errors": report.errors[:50], "news": report.news,
        "appended": report.appended, "predictions": report.predictions, "outcomes": report.outcomes,
        "priors": report.priors, "verify_ok": report.verify["ok"], "backtest": report.backtest.get("added"),
    }
    state.write_cache("status.json", status)
    if log:
        log(f"cycle {report.at}: +{report.predictions} predictions, +{report.outcomes} outcomes, "
            f"news +{report.news.get('new', 0)}, verify {'OK' if report.verify['ok'] else 'FAILED'}")
    return report
