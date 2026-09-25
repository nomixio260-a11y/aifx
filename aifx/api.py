"""JSON API for the web page. The page only reads these documents; it never
computes forecasts itself.

    api/meta.json          status, per-pair summary, verification summary
    api/pair/<PAIR>.json   bars, current forecasts, past forecasts vs actual
    api/track.json         live track record
    api/news.json          analysed headlines, currency pressure, calendar
    api/models.json        learned weights, calibration, walk-forward results, long-history research
    api/market.json        market analysis: currency strength, volatility, trend, upcoming events
    api/backtest.json      rolling backtest on the stored history, per timeframe and horizon
    api/verify.json        full verification and audit reports
"""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, backtest, indicators
from . import news as newsmod
from . import scenario, track, trade
from .rates import latest as rates_latest
from .data import CURRENCIES, PAIRS
from .engine import (BP, FAN_LEVELS, MODEL_KEYS, TIMEFRAMES, bars_until, dist_cdf, dist_pdf, dist_quantile, fan_z,
                     step_ends)
from .forecaster import model_version
from .learning import learn, samples_from_ledger
from .models import default_models
from .pipeline import State
from .timeutil import iso, london_date, parse_iso, utcnow

HISTORY = {"15m": 288, "1h": 240, "1d": 520}
PAST = {"15m": 96, "1h": 96, "1d": 40}
RESEARCH = Path(__file__).resolve().parent.parent / "research" / "results.json"
RESEARCH_ROWS = {
    "1d": [("6モデルの均等平均", "6モデルの均等平均"), ("金利差 (キャリー)", "金利差 (キャリー)"),
           ("モメンタム (過去60本の値動き)", "モメンタム (過去60日)"), ("新設定 (λ=50)", "本番の方式 (学習ルールを再現)")],
    "1h": [("6モデルの均等平均", "6モデルの均等平均"), ("時間帯ごとの平均的な値動き", "時間帯ごとの値動きの癖"),
           ("モメンタム (過去24本の値動き)", "モメンタム (過去24時間)"), ("新設定 (λ=50)", "本番の方式 (学習ルールを再現)")],
}


TRADE_RESEARCH = RESEARCH.parent / "trade.json"
TRADE_BT = {"1d": timedelta(days=365), "1h": timedelta(days=90)}
TRADE_LIST = 40


def trade_research(path: Path = TRADE_RESEARCH) -> dict:
    """Tested numbers of each timeframe's reference rule (research/trade.md)."""
    try:
        res = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for tf_key, rule in trade.RULES.items():
        r = res.get(tf_key, {})
        fam = r.get("families", {}).get(rule["key"])
        if not fam or not fam.get("chosen"):
            continue
        keep = ("n", "win", "pips", "R", "pf", "t", "maxdd_R", "per_year")
        out[tf_key] = {"tier": fam.get("tier"), "start": r.get("start"), "split": r.get("split"), "end": r.get("end"),
                       **{part: {k: _r(fam["chosen"][part].get(k), 4) for k in keep} for part in ("tune", "test")}}
    return out


def _trade_item(tf_key: str, t: dict, dec: int) -> dict:
    res = t.get("result", t)
    out = {"origin": t["origin"], "x": _xkey(tf_key, t["origin"]), "dir": t["dir"], "entry": _r(t["entry"], dec),
           "sl": _r(t["sl"], dec), "tp": _r(t["tp"], dec)}
    if t.get("until"):
        out.update({"until": t["until"], "x_until": _xkey(tf_key, t["until"])})
    if res and res.get("t"):
        out.update({"exit": _r(res["exit"], dec), "how": res["how"], "t": res["t"], "x_exit": _xkey(tf_key, res["t"]),
                    "pips": _r(res["pips"], 1)})
    return out


def _trade_block(tf, pair, rec: dict, bars: pd.DataFrame, ref: pd.DataFrame, preds: list[dict], rate_item: dict | None,
                 now, research: dict) -> dict:
    """The trade plan of the latest forecast, both-side levels, and how the rule has done."""
    dec = pair.decimals + 1
    tr = rec.get("trade")
    recorded = tr is not None
    if tr is None:      # made by an earlier version: the same plan, computed for display only
        tr = trade.plan(tf, pair, bars_until(tf, bars, parse_iso(rec["origin"])).iloc[-tf.fit_bars:],
                        parse_iso(rec["origin"]), rec["p0"], rate_item)
    rule = trade.RULES.get(tf.key)
    out = {"cost_pips": trade.COST_PIPS.get(pair.code), "swap_markup": trade.SWAP_MARKUP,
           "plan": {"origin": rec["origin"], "p0": rec["p0"], "dir": tr.get("dir", 0), "sl": tr.get("sl"), "tp": tr.get("tp"),
                    "until": tr.get("until"), "diff": tr.get("diff"), "atr": tr.get("atr"),
                    "levels": trade.levels(tr, rec["p0"], tf.key, pair.decimals), "recorded": recorded,
                    "x_origin": _xkey(tf.key, rec["origin"]), "x_until": _xkey(tf.key, tr["until"]) if tr.get("until") else None},
           "rule": None}
    if rule is None:
        return out
    out["rule"] = {"key": rule["key"], "name": rule["name"], "desc": rule["desc"], "sl": rule["sl"], "tp": rule["tp"],
                   "hold": rule["hold"], "research": research.get(tf.key)}
    live = trade.live_trades(preds, ref, TIMEFRAMES[tf.ref].minutes, pair)
    done = [dict(t["result"], R=t["result"].get("R")) for t in live if t["result"]]
    out["live"] = {"stats": trade.stats(done), "trades": [_trade_item(tf.key, t, dec) for t in live[-TRADE_LIST:]],
                   "open": next((_trade_item(tf.key, t, dec) for t in reversed(live) if not t["result"]), None)}
    window = bars.iloc[-(int(TRADE_BT[tf.key].days * (24 if tf.key == "1h" else 1)) + 400):]
    bt = trade.backtest(tf, pair, window, rate_item, since=now - TRADE_BT[tf.key])
    out["bt"] = {"days": TRADE_BT[tf.key].days, "stats": trade.stats(bt),
                 "trades": [_trade_item(tf.key, t, dec) for t in bt[-TRADE_LIST:]]}
    return out


CANDLE_EVAL = {"15m": (timedelta(days=3), 5), "1h": (timedelta(days=20), 7), "1d": (timedelta(days=365), 5)}
CANDLE_RESEARCH = RESEARCH.parent / "candles.json"


def candle_eval(tf, bars: pd.DataFrame, now) -> list[tuple]:
    """Forecast candles rebuilt at recent origins from the bars up to each origin, against the real
    candles that followed: (step, forecast dir, actual dir, forecast colour, actual colour,
    forecast size, actual size, 14-bar average size)."""
    window, every = CANDLE_EVAL[tf.key]
    steps = max(tf.horizons)
    if len(bars) < 400:
        return []
    o, h, lo, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = pd.Series(h - lo).rolling(14).mean().to_numpy()
    since = pd.Timestamp(now - window)
    if bars.index.tz is None:
        since = since.tz_localize(None)
    out = []
    first = int(bars.index.searchsorted(since))
    for t in range(max(first, 300), len(c) - 1, every):
        cs, _ = scenario.candles(tf.key, bars.iloc[: t + 1], steps, float(c[t]), tf.minutes)
        for j, (po, ph, pl, pc) in enumerate(cs):
            k = t + 1 + j
            if k >= len(c):
                break
            out.append((j + 1, np.sign(pc - c[t]), np.sign(c[k] - c[t]), np.sign(pc - po), np.sign(c[k] - o[k]),
                        ph - pl, h[k] - lo[k], atr[t]))
    return out


def candle_stats(rows: list[tuple]) -> dict:
    if not rows:
        return {"n": 0}
    R = np.array(rows, dtype=float)
    d = (R[:, 1] != 0) & (R[:, 2] != 0)
    b = (R[:, 3] != 0) & (R[:, 4] != 0)
    ok = np.isfinite(R[:, 7])
    mae = np.mean(np.abs(R[ok, 5] - R[ok, 6]))
    mae_atr = np.mean(np.abs(R[ok, 7] - R[ok, 6]))
    return {"n": int(len(R)), "dir_hit": _r(float(np.mean(R[d, 1] == R[d, 2])), 4) if d.any() else None,
            "color_hit": _r(float(np.mean(R[b, 3] == R[b, 4])), 4) if b.any() else None,
            "size_vs_atr": _r(float(mae / mae_atr - 1), 4) if mae_atr > 0 else None}


def research_summary(path: Path = RESEARCH) -> dict | None:
    """Headline numbers of the long-history research (research/report.md), if present."""
    try:
        res = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    out = {"report": "research/report.md", "tf": {}}
    for tf in ("1d", "1h"):
        r = res[tf]
        rows = []
        for key, label in RESEARCH_ROWS[tf]:
            src = r["point"].get(key) or r["sim"].get(key)
            if src is None:
                continue
            rows.append({"name": label, "h": {h: {"skill": _r(x["skill"], 5), "hit": _r(x["hit"], 4), "p": _r(x["dm_p"], 4)}
                                             for h, x in src["test"].items()},
                         "tune": {h: {"skill": _r(x["skill"], 5), "hit": _r(x["hit"], 4)} for h, x in src["tune"].items()}})
        ranges = {h: {"before": {k: _r(v, 4) for k, v in r["shape"]["before"][h]["normal"]["cover"].items()},
                      "after": {k: _r(v, 4) for k, v in r["shape"]["after"][h]["t"]["cover"].items()}}
                  for h in r["shape"]["before"]}
        out["tf"][tf] = {"tune": r["tune"], "test": r["test"], "n": r["n"], "direction": rows, "ranges": ranges}
    intra = path.parent / "intraday.json"
    try:
        m = json.loads(intra.read_text(encoding="utf-8"))["15m"]
    except (OSError, ValueError, KeyError):
        return out
    rows = []
    for key, label in (("6モデルの均等平均", "6モデルの均等平均"), ("直前15分の値動き (反転/継続)", "直前15分の値動き"),
                       (next((k for k in m["point"] if k.startswith("本番の学習ルール")), ""), "本番の方式 (学習ルールを再現)")):
        src = m["point"].get(key)
        if src:
            rows.append({"name": label, "h": {h: {"skill": _r(x["skill"], 5), "hit": _r(x["hit"], 4), "p": _r(x["dm_p"], 4)}
                                             for h, x in src["test"].items()}})
    ranges = {h: {"before": {k: _r(v, 4) for k, v in m["shape"]["before"][h]["normal"]["cover"].items()},
                  "after": {k: _r(v, 4) for k, v in m["shape"]["after"][h]["t"]["cover"].items()}}
              for h in m["shape"]["before"]}
    out["tf"]["15m"] = {"tune": m["tune"], "test": m["test"], "n": m["n"], "direction": rows, "ranges": ranges,
                        "report": "research/intraday.md"}
    return out


def _r(v, d):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return round(float(v), d)


def _label(p: float) -> str:
    return "上昇" if p >= 0.55 else "下落" if p <= 0.45 else "横ばい"


def _xkey(tf: str, t: str) -> str:
    """Chart x-axis key for an observation time: when the intraday bar closes, or its London day (1d)."""
    if TIMEFRAMES[tf].minutes:
        return t
    return london_date(parse_iso(t) - timedelta(minutes=1)).strftime("%Y-%m-%d")


def _bars(df: pd.DataFrame, n: int, tf: str, dec: int) -> dict:
    d = df.iloc[-n:]
    # Intraday bars are keyed by their closing time so they line up with forecast targets.
    minutes = TIMEFRAMES[tf].minutes
    t = [iso(x.to_pydatetime() + timedelta(minutes=minutes)) for x in d.index] if minutes else [
        x.strftime("%Y-%m-%d") for x in d.index]
    ohlc = [[_r(o, dec), _r(c, dec), _r(lo, dec), _r(hi, dec)]
            for o, hi, lo, c in d[["open", "high", "low", "close"]].to_numpy()]
    return {"t": t, "ohlc": ohlc}


def _indicators(df: pd.DataFrame, n: int, dec: int) -> dict:
    ind = indicators.compute_all(df).iloc[-n:]
    return {k: [_r(v, dec if k != "rsi14" else 2) for v in ind[k].to_numpy()] for k in
            ("sma20", "sma75", "bb_upper", "bb_lower", "rsi14")}


QUANTILES = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975)
LEVEL_STEPS_PIPS = (5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)


def distribution(p0: float, c: float, sig: float, nu, pair) -> dict:
    """The forecast as a price distribution: quantiles, a density curve and the chance of
    finishing above round price levels. Everything is computed here, on the server."""
    dec = pair.decimals + 1

    def price(z):
        return p0 * math.exp((c + z * sig) / BP)

    q = {f"{t:g}": _r(price(dist_quantile(t, nu)), dec) for t in QUANTILES}
    zs = np.linspace(dist_quantile(0.004, nu), dist_quantile(0.996, nu), 73)
    prices = np.array([price(z) for z in zs])
    dens = dist_pdf(zs, nu) / (prices * sig / BP)          # per unit of price
    dens = dens / dens.max()
    curve = [[_r(pv, dec), _r(dv, 4), _r(1 - dist_cdf(float(z), nu), 4)] for pv, dv, z in zip(prices, dens, zs)]
    lo, hi = price(dist_quantile(0.025, nu)), price(dist_quantile(0.975, nu))
    span_pips = (hi - lo) / pair.pip
    step = next((s for s in LEVEL_STEPS_PIPS if span_pips / s <= 9), LEVEL_STEPS_PIPS[-1]) * pair.pip
    levels = []
    lv = math.ceil(lo / step) * step
    while lv <= hi + 1e-12:
        z = (math.log(lv / p0) * BP - c) / sig if sig > 0 else 0.0
        levels.append({"price": _r(lv, pair.decimals), "p_above": _r(1 - dist_cdf(z, nu), 4)})
        lv += step
    return {"q": q, "curve": curve, "levels": levels[::-1], "step_pips": _r(step / pair.pip, 1)}


def _horizons(rec: dict, pair) -> list[dict]:
    out = []
    p0 = rec["p0"]
    for f in rec["fc"]:
        sig = f["s"] * f["k"]
        price = p0 * math.exp(f["c"] / BP)
        z = fan_z(f.get("nu"))
        out.append({
            "h": f["h"], "label": TIMEFRAMES[rec["tf"]].horizon_label(f["h"]), "t": f["t"], "x": _xkey(rec["tf"], f["t"]),
            "price": _r(price, pair.decimals + 1),
            "change_pips": _r((price - p0) / pair.pip, 1),
            "change_pct": _r((math.exp(f["c"] / BP) - 1) * 100, 3),
            "p_up": _r(f["p"], 4), "dir": _label(f["p"]),
            **{f"{side}{lv}": _r(p0 * math.exp((f["c"] + sgn * z[lv] * sig) / BP), pair.decimals + 1)
               for lv in FAN_LEVELS for side, sgn in (("lo", -1), ("hi", 1))},
            "dist": distribution(p0, f["c"], sig, f.get("nu"), pair),
            "models_up": sum(1 for v in f["m"][1:] if v > 0), "models_total": len(f["m"]) - 1,
            "news_pips": _r(p0 * (math.exp(f["c"] / BP) - math.exp(f["g"] * f["c0"] / BP)) / pair.pip, 2),
            "events": f["ev"], "k": f["k"], "beta": f["b"], "gain": f["g"],
        })
    return out


def _coarse_path(rec: dict, horizons: list[dict], tf_key: str, dec: int) -> dict:
    """Chart path when the per-step detail cannot be rebuilt (the forecasting code changed
    since the prediction): the recorded horizons, joined by straight lines for display."""
    tf = TIMEFRAMES[tf_key]
    origin = parse_iso(rec["origin"])
    ends = step_ends(tf, origin, max(tf.steps, max(tf.horizons)))
    keys = ["c"] + [f"{side}{lv}" for lv in FAN_LEVELS for side in ("lo", "hi")
                    if all(f"{side}{lv}" in h for h in horizons)]
    anchors = [(0, {k: rec["p0"] for k in keys} | {"p": 0.5})]
    for h in horizons:
        anchors.append((h["h"], {"c": h["price"], "p": h["p_up"], **{k: h[k] for k in keys[1:]}}))
    steps = []
    for i, end in enumerate(ends, start=1):
        a = max((x for x in anchors if x[0] <= i), key=lambda x: x[0])
        b = min((x for x in anchors if x[0] >= i), key=lambda x: x[0], default=a)
        w = 0.0 if b[0] == a[0] else (i - a[0]) / (b[0] - a[0])
        row = {k: _r(a[1][k] + w * (b[1][k] - a[1][k]), dec) for k in keys}
        row["p"] = _r(a[1]["p"] + w * (b[1]["p"] - a[1]["p"]), 4)
        steps.append({"t": iso(end), "x": _xkey(tf_key, iso(end)), **row})
    return {"steps": steps, "models": {}, "events": [], "news": None, "analogs": [], "coarse": True}


def _past(rows: list[dict], pair: str, tf: str, n: int, dec: int) -> dict[str, list[dict]]:
    """The latest scored live forecasts of each horizon, for the answer-check overlay."""
    out: dict[str, list[dict]] = {}
    for h in TIMEFRAMES[tf].horizons:
        sel = [r for r in rows if r["pair"] == pair and r["tf"] == tf and r["h"] == h][-n:]
        items = []
        for r in sel:
            pred = r["p0"] * math.exp(r["c"] / BP)
            moved = abs(r["a"]) > 1e-9 and abs(r["c"]) > 1e-9
            items.append({"t": r["target"], "x": _xkey(tf, r["target"]), "origin": r["origin"], "price": _r(pred, dec),
                          **{f"{side}{lv}": _r(r["p0"] * math.exp((r["c"] + sgn * r["z"][lv] * r["sigma"]) / BP), dec)
                             for lv in ("50", "80", "95") for side, sgn in (("lo", -1), ("hi", 1))},
                          "actual": _r(r["actual"], dec), "hit": ((r["c"] > 0) == (r["a"] > 0)) if moved else None,
                          "inside80": abs(r["c"] - r["a"]) <= r["z"]["80"] * r["sigma"]})
        out[str(h)] = items
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
    # The last ledger record is the authoritative time of the last server cycle.
    cycle_at = status.get("at") or (ledger.records[-1]["at"] if ledger.records else None)
    now = parse_iso(cycle_at) if cycle_at else utcnow()
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
    rate_item = rates_latest(state.rates.load(since=now - timedelta(days=40)))
    candle_rows: dict[str, list] = {}
    trade_res = trade_research()
    preds_by = {}
    for p in preds:
        preds_by.setdefault((p["pair"], p["tf"]), []).append(p)
    ranges24: dict[str, dict] = {}
    hourly_all: dict[str, pd.DataFrame] = {}
    daily_all: dict[str, pd.DataFrame] = {}
    for code, pair in PAIRS.items():
        dec = pair.decimals + 1
        hourly = state.prices.load(code, "1h")
        daily = state.prices.load(code, "1d")
        if not len(hourly):
            continue
        hourly_all[code], daily_all[code] = hourly, daily
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
            bars = hourly if tf_key == "1h" else daily if tf_key == "1d" else state.prices.load(code, tf_key)
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
                summary["outlook"][tf_key] = [{k: h[k] for k in ("h", "label", "price", "p_up", "dir", "change_pips",
                                                                 "lo80", "hi80")}
                                              for h in block["prediction"]["horizons"]]
                ref_bars = hourly if tf.ref == "1h" else state.prices.load(code, tf.ref)
                block["trade"] = _trade_block(tf, pair, rec, bars, ref_bars, preds_by.get((code, tf_key), []), rate_item,
                                              now, trade_res)
                summary.setdefault("signal", {})[tf_key] = block["trade"]["plan"]["dir"] if block["trade"]["rule"] else None
                if tf_key == "1h":
                    h24 = block["prediction"]["horizons"][-1]
                    ranges24[code] = {"h": h24["h"], "lo80": h24["lo80"], "hi80": h24["hi80"],
                                      "half80_pips": _r((h24["hi80"] - h24["lo80"]) / 2 / pair.pip, 1)}
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
            elif rec is not None:
                block["path"] = _coarse_path(rec, block["prediction"]["horizons"], tf_key, dec)
            if rec is not None and block.get("path"):
                steps_ = block["path"]["steps"]
                hist = bars_until(tf, bars, parse_iso(rec["origin"]))
                cs, info = scenario.candles(tf_key, hist, len(steps_), steps_[-1]["c"], tf.minutes)
                if cs:
                    block["candles"] = {"items": [[_r(v, dec) for v in k] for k in cs], "x": [st["x"] for st in steps_],
                                        "t": [st["t"] for st in steps_], "analog": str(info["analog_end"])}
                candle_rows.setdefault(tf_key, []).extend(candle_eval(tf, bars, now))
            if tf_key == "1d" and len(daily) > 80:
                block["technical"] = indicators.technical_summary(daily, indicators.compute_all(daily))
            payload["tf"][tf_key] = block
        payloads[code] = payload
        pair_summaries.append(summary)

    # forecast candles: how they have done lately (all pairs), and in the research
    try:
        cres = json.loads(CANDLE_RESEARCH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cres = {}
    for tf_key, rows_ in candle_rows.items():
        steps_all = max(TIMEFRAMES[tf_key].horizons)
        acc = {"window_days": CANDLE_EVAL[tf_key][0].days or 1, "all": candle_stats(rows_),
               "next": candle_stats([r for r in rows_ if r[0] == 1]), "last": candle_stats([r for r in rows_ if r[0] == steps_all]),
               "research": (cres.get(tf_key) or {}).get("h")}
        for code in payloads:
            blk = payloads[code]["tf"].get(tf_key)
            if blk and "candles" in blk:
                blk["candles"]["accuracy"] = acc

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
    story_of = newsmod.stories(recent_items)
    copies: dict[int, int] = {}
    for sid in story_of.values():
        copies[sid] = copies.get(sid, 0) + 1
    out["news.json"] = {
        "at": iso(now),
        "analyzer": newsmod.ANALYZER,
        "pressures": press_now,
        "pair_signals": {c: newsmod.pair_signal(press_now, p.base, p.quote) for c, p in PAIRS.items()},
        "history": history,
        "items": [{k: it.get(k) for k in ("id", "title", "publisher", "link", "published_at", "fetched_at", "lang", "src")}
                  | {"cur": it["an"]["cur"], "top": it["an"]["top"], "by": it["an"]["by"], "ja": it["an"].get("ja"),
                     "story": story_of.get(it["id"]), "copies": copies.get(story_of.get(it["id"]), 1)}
                  for it in recent_items],
        "calendar": cal,
        "sources": (status.get("news") or {}).get("sources", {}),
        "params": {"tau_hours": newsmod.TAU_HOURS, "lookback_hours": newsmod.LOOKBACK_HOURS, "shrink": newsmod.SHRINK},
    }

    # rolling backtest (derived from the stored prices; see backtest.py) ------
    bt = backtest.summary(state, list(TIMEFRAMES.values()), list(payloads), now, {c: PAIRS[c].decimals for c in payloads})
    for code, payload in payloads.items():
        for tf_key, block in payload["tf"].items():
            per_h = bt["pairs"].get(code, {}).get(tf_key, {})
            for items in per_h.values():
                for it in items:
                    it["x"] = _xkey(tf_key, it["t"])
            block["bt_past"] = per_h
    out["backtest.json"] = {"at": bt["at"], "tf": bt["tf"]}

    # market analysis (descriptive) -----------------------------------------
    out["market.json"] = analysis.build(hourly_all, daily_all, events, press_now, ranges24, now)

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
        "research": research_summary(),
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
        "cycle_at": cycle_at,
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
