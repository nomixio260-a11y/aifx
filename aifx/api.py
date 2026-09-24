"""JSON API for the web page. The page only reads these documents; it never
computes forecasts itself.

    api/meta.json          status, per-pair summary, verification summary
    api/pair/<PAIR>.json   bars, current forecasts, past forecasts vs actual
    api/track.json         live track record
    api/news.json          analysed headlines, currency pressure, calendar
    api/models.json        learned weights, calibration, walk-forward results
    api/verify.json        full verification and audit reports
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import indicators
from . import news as newsmod
from . import track
from .data import CURRENCIES, PAIRS
from .engine import BAND_Z, BP, MODEL_KEYS, TIMEFRAMES
from .forecaster import model_version
from .learning import learn, samples_from_ledger
from .models import default_models
from .pipeline import State
from .timeutil import iso, london_date, parse_iso, utcnow

HISTORY = {"1h": 240, "1d": 520}
PAST = {"1h": 96, "1d": 40}


def _r(v, d):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), d)


def _label(p: float) -> str:
    return "上昇" if p >= 0.55 else "下落" if p <= 0.45 else "横ばい"


def _xkey(tf: str, t: str) -> str:
    """Chart x-axis key for an observation time: the hour it closes (1h) or its London day (1d)."""
    if tf == "1h":
        return t
    return london_date(parse_iso(t) - timedelta(minutes=1)).strftime("%Y-%m-%d")


def _bars(df: pd.DataFrame, n: int, tf: str, dec: int) -> dict:
    d = df.iloc[-n:]
    # Hourly bars are keyed by their closing time so they line up with forecast targets.
    t = [iso(x.to_pydatetime() + timedelta(hours=1)) for x in d.index] if tf == "1h" else [
        x.strftime("%Y-%m-%d") for x in d.index]
    ohlc = [[_r(o, dec), _r(c, dec), _r(lo, dec), _r(hi, dec)]
            for o, hi, lo, c in d[["open", "high", "low", "close"]].to_numpy()]
    return {"t": t, "ohlc": ohlc}


def _indicators(df: pd.DataFrame, n: int, dec: int) -> dict:
    ind = indicators.compute_all(df).iloc[-n:]
    return {k: [_r(v, dec if k != "rsi14" else 2) for v in ind[k].to_numpy()] for k in
            ("sma20", "sma75", "bb_upper", "bb_lower", "rsi14")}


def _horizons(rec: dict, pair) -> list[dict]:
    out = []
    p0 = rec["p0"]
    for f in rec["fc"]:
        sig = f["s"] * f["k"]
        price = p0 * math.exp(f["c"] / BP)
        out.append({
            "h": f["h"], "label": TIMEFRAMES[rec["tf"]].horizon_label(f["h"]), "t": f["t"], "x": _xkey(rec["tf"], f["t"]),
            "price": _r(price, pair.decimals + 1),
            "change_pips": _r((price - p0) / pair.pip, 1),
            "change_pct": _r((math.exp(f["c"] / BP) - 1) * 100, 3),
            "p_up": _r(f["p"], 4), "dir": _label(f["p"]),
            "lo80": _r(p0 * math.exp((f["c"] - BAND_Z["80"] * sig) / BP), pair.decimals + 1),
            "hi80": _r(p0 * math.exp((f["c"] + BAND_Z["80"] * sig) / BP), pair.decimals + 1),
            "models_up": sum(1 for v in f["m"][1:] if v > 0), "models_total": len(f["m"]) - 1,
            "news_pips": _r(p0 * (math.exp(f["c"] / BP) - math.exp(f["g"] * f["c0"] / BP)) / pair.pip, 2),
            "events": f["ev"], "k": f["k"], "beta": f["b"], "gain": f["g"],
        })
    return out


def _past(rows: list[dict], pair: str, tf: str, n: int, dec: int) -> list[dict]:
    sel = [r for r in rows if r["pair"] == pair and r["tf"] == tf and r["h"] == 1][-n:]
    out = []
    for r in sel:
        pred = r["p0"] * math.exp(r["c"] / BP)
        moved = abs(r["a"]) > 1e-9 and abs(r["c"]) > 1e-9
        out.append({"t": r["target"], "x": _xkey(tf, r["target"]), "origin": r["origin"], "price": _r(pred, dec),
                    "lo80": _r(r["p0"] * math.exp((r["c"] - BAND_Z["80"] * r["sigma"]) / BP), dec),
                    "hi80": _r(r["p0"] * math.exp((r["c"] + BAND_Z["80"] * r["sigma"]) / BP), dec),
                    "actual": _r(r["actual"], dec), "hit": ((r["c"] > 0) == (r["a"] > 0)) if moved else None,
                    "inside80": abs(r["c"] - r["a"]) <= BAND_Z["80"] * r["sigma"]})
    return out


def _charts(state, ledger, preds, outcomes, last_pred, cached: dict) -> dict:
    """Per-step chart paths for the latest predictions, rebuilt from the ledger if not cached."""
    from .audit import _Committed, reconstruct
    out = {}
    com = None
    pred_map = {p["seq"]: p for p in preds}
    for (code, tf_key), rec in last_pred.items():
        chart = (cached.get(code) or {}).get(tf_key)
        if chart is None or chart.get("seq") != rec["seq"]:
            if rec["v"] != model_version():
                continue
            com = com or _Committed(state.root, ledger)
            _, chart = reconstruct(ledger, com, rec, pred_map, outcomes)
            chart["seq"] = rec["seq"]
        out.setdefault(code, {})[tf_key] = chart
    return out


def _learning(ledger, preds, outcomes) -> dict:
    pred_map = {p["seq"]: p for p in preds}
    out = {}
    for tf_key, tf in TIMEFRAMES.items():
        prior = next((r for r in reversed(ledger.records) if r["type"] == "prior" and r["tf"] == tf_key), None)
        if prior is None:
            continue
        st = learn(prior, samples_from_ledger(pred_map, outcomes, tf_key), tf.horizons, tf.half_life)
        out[tf_key] = {str(h): s.as_dict() for h, s in st.items()}
    return out


def build_api(root: Path | str, mode: str = "static", interval_min: float = 15) -> dict[str, dict]:
    state = State.open(root)
    ledger = state.ledger
    preds = ledger.of_type("prediction")
    outcomes = ledger.of_type("outcome")
    status = state.read_cache("status.json", {}) or {}
    latest = state.read_cache("latest.json", {}) or {}
    audit_rep = state.read_cache("audit.json", None)
    now = parse_iso(status["at"]) if status.get("at") else utcnow()
    pip_of = {c: p.pip for c, p in PAIRS.items()}
    tr = track.build(preds, outcomes, pip_of)
    rows, _ = track.join(preds, outcomes)
    last_pred = {}
    for p in preds:
        last_pred[(p["pair"], p["tf"])] = p
    latest = _charts(state, ledger, preds, outcomes, last_pred, latest)
    learning = _learning(ledger, preds, outcomes)

    out: dict[str, dict] = {}
    payloads: dict[str, dict] = {}
    pair_summaries = []
    for code, pair in PAIRS.items():
        dec = pair.decimals + 1
        hourly = state.prices.load(code, "1h")
        daily = state.prices.load(code, "1d")
        if not len(hourly):
            continue
        quote = status.get("quotes", {}).get(code) or {}
        last_close = float(hourly["close"].iloc[-1])
        price = float(quote.get("price") or last_close)
        ref = float(hourly["close"].iloc[-25]) if len(hourly) > 25 else last_close
        payload = {
            "pair": code, "label": pair.label, "name": pair.name, "decimals": pair.decimals, "pip": pair.pip,
            "quote": {"price": _r(price, dec), "time": iso(pd.Timestamp(quote["time"], unit="s", tz="UTC").to_pydatetime())
                      if quote.get("time") else iso(hourly.index[-1].to_pydatetime() + timedelta(hours=1))},
            "last_bar": iso(hourly.index[-1].to_pydatetime() + timedelta(hours=1)),
            "change_24h_pct": _r((last_close / ref - 1) * 100, 3),
            "tf": {},
        }
        summary = {"pair": code, "label": pair.label, "name": pair.name, "decimals": pair.decimals, "pip": pair.pip,
                   "price": payload["quote"]["price"], "change_24h_pct": payload["change_24h_pct"], "outlook": {},
                   "live": tr["by_pair"].get(code, {}).get("1h", {}).get("1", {})}
        for tf_key, tf in TIMEFRAMES.items():
            bars = hourly if tf_key == "1h" else daily
            if not len(bars):
                continue
            block = {
                "bars": _bars(bars, HISTORY[tf_key], tf_key, dec),
                "ind": _indicators(bars, HISTORY[tf_key], dec),
                "past": _past(rows, code, tf_key, PAST[tf_key], dec),
            }
            rec = last_pred.get((code, tf_key))
            chart = latest.get(code, {}).get(tf_key)
            if rec is not None:
                block["prediction"] = {"seq": rec["seq"], "issued": rec["at"], "origin": rec["origin"], "p0": rec["p0"],
                                       "news_x": rec["news"]["x"], "horizons": _horizons(rec, pair)}
                summary["outlook"][tf_key] = [{k: h[k] for k in ("h", "label", "price", "p_up", "dir", "change_pips")}
                                              for h in block["prediction"]["horizons"]]
            if chart is not None and rec is not None and chart.get("seq") == rec["seq"]:
                steps = [{"t": st["t"], "x": _xkey(tf_key, st["t"]), "p": _r(st["p"], 4),
                          **{k: _r(v, dec) for k, v in st.items() if k not in ("t", "p")}} for st in chart["steps"]]
                evs = []
                for ev in chart["events"]:
                    step = next((st for st in steps if st["t"] >= ev["time"]), None)
                    evs.append({**ev, "x": step["x"] if step else None})
                block["path"] = {
                    "steps": steps,
                    "models": {k: [_r(v, dec) for v in vals] for k, vals in chart["models"].items()},
                    "events": evs, "news": chart["news"], "analogs": chart["analogs"],
                }
            if tf_key == "1d" and len(daily) > 80:
                block["technical"] = indicators.technical_summary(daily, indicators.compute_all(daily))
            payload["tf"][tf_key] = block
        payloads[code] = payload
        pair_summaries.append(summary)

    # news --------------------------------------------------------------
    items = state.news.load(since=now - timedelta(days=4))
    display_cut = now + timedelta(seconds=1)
    recent_items = sorted([it for it in items if it["published_at"] >= iso(now - timedelta(hours=48))],
                          key=lambda it: it["published_at"], reverse=True)[:150]
    press_now = newsmod.pressures(items, display_cut)
    history = {c: [] for c in CURRENCIES}
    # Display only: how the news flow evolved by publication time. Forecasts use the
    # stricter rule (headline must have been stored before the origin).
    by_published = [{**it, "fetched_at": ""} for it in items]
    for k in range(48, -1, -2):
        cut = now - timedelta(hours=k)
        pr = newsmod.pressures(by_published, cut)
        for c in CURRENCIES:
            history[c].append([iso(cut), pr[c]["p"]])
    events = state.calendar.load(since=now - timedelta(days=40))
    for code, payload in payloads.items():
        pair = PAIRS[code]
        mine = [it for it in recent_items if pair.base in it["an"]["cur"] or pair.quote in it["an"]["cur"]]
        payload["headlines"] = [{"title": it["title"], "publisher": it["publisher"], "link": it["link"],
                                 "published_at": it["published_at"], "cur": it["an"]["cur"], "by": it["an"]["by"],
                                 "ja": it["an"].get("ja")} for it in mine[:8]]
        payload["calendar"] = sorted([e for e in events if e["cur"] in (pair.base, pair.quote)
                                      and iso(now) <= e["time"] <= iso(now + timedelta(days=7))],
                                     key=lambda e: e["time"])[:12]
        payload["pressure"] = {pair.base: press_now[pair.base], pair.quote: press_now[pair.quote],
                               "signal": newsmod.pair_signal(press_now, pair.base, pair.quote)}
        out[f"pair/{code}.json"] = payload
    cal = sorted([e for e in events if iso(now) <= e["time"] <= iso(now + timedelta(days=7))],
                 key=lambda e: e["time"])
    out["news.json"] = {
        "at": iso(now),
        "analyzer": (status.get("news") or {}).get("analyzer", newsmod.ANALYZER),
        "llm_usage": (status.get("news") or {}).get("llm_usage"),
        "pressures": press_now,
        "pair_signals": {c: newsmod.pair_signal(press_now, p.base, p.quote) for c, p in PAIRS.items()},
        "history": history,
        "items": [{k: it.get(k) for k in ("id", "title", "publisher", "link", "published_at", "fetched_at", "lang", "src")}
                  | {"cur": it["an"]["cur"], "top": it["an"]["top"], "by": it["an"]["by"], "ja": it["an"].get("ja")}
                  for it in recent_items],
        "calendar": cal,
        "sources": (status.get("news") or {}).get("sources", {}),
        "params": {"tau_hours": newsmod.TAU_HOURS, "lookback_hours": newsmod.LOOKBACK_HOURS, "shrink": newsmod.SHRINK},
    }

    # models and learning -------------------------------------------------
    priors = {}
    for rec in ledger.of_type("prior"):
        priors[rec["tf"]] = {"seq": rec["seq"], "at": rec["at"], "cutoff": rec["cutoff"], "h": rec["h"]}
    out["models.json"] = {
        "models": [{"key": m.key, "name": m.name, "description": m.description} for m in default_models()],
        "learning": learning,
        "priors": priors,
        "timeframes": {k: {"label": tf.label, "horizons": list(tf.horizons), "half_life": tf.half_life}
                       for k, tf in TIMEFRAMES.items()},
    }

    # track record ----------------------------------------------------------
    out["track.json"] = {"at": iso(now), **tr}

    # verification --------------------------------------------------------------
    from .audit import verify
    ver = verify(state.root)
    out["verify.json"] = {"verify": ver, "audit": audit_rep, "external": state.read_cache("external.json", None)}

    first = preds[0]["at"] if preds else None
    out["meta.json"] = {
        "generated_at": iso(utcnow()),
        "cycle_at": status.get("at"),
        "mode": mode,
        "interval_min": interval_min,
        "next_update_at": iso(now + timedelta(minutes=interval_min)),
        "pairs": pair_summaries,
        "models": [{"key": m.key, "name": m.name} for m in default_models()] + [{"key": "ensemble", "name": "アンサンブル"}],
        "timeframes": {k: {"label": tf.label, "horizons": list(tf.horizons),
                           "labels": [tf.horizon_label(h) for h in tf.horizons]} for k, tf in TIMEFRAMES.items()},
        "ledger": {"records": ver["records"], "head": ver["head"], "ok": ver["ok"], "n_problems": ver["n_problems"],
                   "counts": ver["counts"], "first_prediction": first,
                   "audit_ok": audit_rep.get("ok") if audit_rep else None},
        "track": {tf: {h: {"n": s.get("n", 0), "direction": s.get("direction"), "skill": s.get("skill"),
                           "coverage80": (s.get("coverage") or {}).get("80")}
                       for h, s in tr["overall"][tf].items()} for tf in TIMEFRAMES},
        "news": {"analyzer": out["news.json"]["analyzer"], "items_48h": len(recent_items)},
        "errors": status.get("errors", [])[:20],
        "repo": "https://github.com/nomixio260-a11y/aifx",
    }
    return out


def write_api(files: dict[str, dict], site_dir: Path | str) -> None:
    base = Path(site_dir) / "api"
    for name, obj in files.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        tmp.replace(path)


__all__ = ["build_api", "write_api", "np", "MODEL_KEYS"]
