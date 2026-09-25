"""Rolling walk-forward backtest over the stored history, updated every cycle.

For every timeframe and pair, the forecaster is re-run at every recent origin
using only the bars that had completed by that origin (the same models,
volatility model and band shape as the live forecaster; no news, because
historical headlines were not stored point-in-time), and each forecast is
compared with what happened. The live learning rule is replayed through time
as well, from the same walk-forward prior: gain, model weights and the range
scale at each origin come only from backtest forecasts whose outcome was
already known then.

This is a hypothetical test of today's code on the stored past. It is shown
next to the live track record, which consists of forecasts recorded in the
ledger before their outcome and can't be re-run with hindsight. Results are
cached in ``cache/backtest/`` (derived data, never committed) and extended
with the new origins every cycle.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .engine import (BP, MODEL_KEYS, TIMEFRAMES, Timeframe, band_nu, band_z, bar_end, bars_until, horizon_sigma,
                     model_paths, prob_up, sigma_steps, target_times)
from . import season
from .forecaster import model_version, origin_price
from .learning import learn_arrays
from .timeutil import iso, parse_iso

WINDOW = {"15m": timedelta(days=10), "1h": timedelta(days=60), "1d": timedelta(days=365)}
RECENT = {"15m": timedelta(hours=24), "1h": timedelta(days=7), "1d": timedelta(days=60)}
# timeline: bucket, buckets in the trailing window (empty ones skipped), and its name
ROLL = {"15m": ("4h", 6, "24時間"), "1h": ("D", 5, "5営業日"), "1d": ("W", 4, "4週間")}
BUDGET = 6000          # origins computed per cycle at most (newest first); the rest follow next cycle
LEVELS = ("50", "80", "95")


def _key(tf: str, pair: str) -> str:
    return f"backtest/{tf}_{pair}.json"


def _version() -> str:
    """The model version plus the code that makes the stored rows: a change to either rebuilds them."""
    code = inspect.getsource(_forecast) + inspect.getsource(_fill_outcomes)
    return f"{model_version()}-{hashlib.sha256(code.encode()).hexdigest()[:8]}"


def update(state, tfs: list[Timeframe], pairs: list[str], now: datetime, budget: int = BUDGET, log=None) -> dict:
    """Add backtest forecasts for new origins and fill in outcomes that are now known.

    The newest missing origins come first, one pair at a time in turn, and the
    budget is shared, so every timeframe and every pair advances each cycle."""
    version = _version()
    added = 0
    report = {}
    for n_tf, tf in enumerate(tfs):
        limit = added + (budget - added) // (len(tfs) - n_tf)
        since = now - WINDOW[tf.key]
        jobs = {}
        for code in pairs:
            cache = state.read_cache(_key(tf.key, code), None)
            if not cache or cache.get("v") != version:
                cache = {"v": version, "rows": {}}
            bars = bars_until(tf, state.prices.load(code, tf.key), now)
            ref = state.prices.load(code, tf.ref)
            if len(bars) < 450 or not len(ref):
                continue
            ends = [bar_end(tf, ts) for ts in bars.index]
            todo = [o for o in range(400, len(bars)) if ends[o] >= since and iso(ends[o]) not in cache["rows"]]
            jobs[code] = (cache, bars, ref, ends, todo[::-1])
        for i in range(max((len(j[4]) for j in jobs.values()), default=0)):
            for cache, bars, ref, ends, todo in jobs.values():
                if added < limit and i < len(todo) and _forecast(tf, cache["rows"], bars, ref, ends[todo[i]], todo[i]):
                    added += 1
            if added >= limit:
                break
        for code, (cache, _, ref, _, _) in jobs.items():
            _fill_outcomes(cache["rows"], ref, TIMEFRAMES[tf.ref].minutes)
            cache["rows"] = {k: v for k, v in cache["rows"].items() if k >= iso(since)}
            state.write_cache(_key(tf.key, code), cache)
            report[f"{code}_{tf.key}"] = len(cache["rows"])
    if log:
        log(f"backtest: +{added} origins")
    return {"added": added, "rows": report}


def _forecast(tf: Timeframe, rows: dict, bars: pd.DataFrame, ref: pd.DataFrame, origin: datetime, o: int) -> bool:
    """The forecast the models would have made at ``origin`` from the bars completed by then
    (the time-of-day drift uses bars from before the origin's day only)."""
    ref_min = TIMEFRAMES[tf.ref].minutes
    ref_o = ref.iloc[:int((ref.index + pd.Timedelta(minutes=ref_min)).searchsorted(pd.Timestamp(origin), side="right"))]
    p = origin_price(ref_o, origin, ref_min)
    if p is None:
        return False
    H = max(tf.horizons)
    hist = bars.iloc[max(0, o + 1 - tf.fit_bars): o + 1]
    paths, _ = model_paths(np.log(hist["close"].to_numpy()), H)
    var = sigma_steps(tf, hist, origin, H, None, hourly=ref_o if tf.key == "1d" else None)
    sig = horizon_sigma(var, tf.horizons)
    targets = target_times(tf, origin)
    bar_d, bar_t = season.step_drift(tf.minutes, bars, origin, H)
    drift, drift_t = season.centre_drift(bar_d, tf.minutes), season.centre_t(bar_d, bar_t, tf.minutes)
    rows[iso(origin)] = {"p0": p[0], "h": {
        str(h): {"t": iso(targets[j]), "m": [round(float(paths[k][h - 1]), 3) for k in MODEL_KEYS],
                 "s": round(sig[j], 4), "d": round(float(drift[h - 1]), 4), "dt": round(float(drift_t[h - 1]), 2),
                 "a": None}
        for j, h in enumerate(tf.horizons)}}
    return True


def _fill_outcomes(rows: dict, ref: pd.DataFrame, ref_min: int) -> None:
    """Outcomes that are known now: the reference bar closing at each target."""
    r_ends = ref.index + pd.Timedelta(minutes=ref_min)
    if not len(r_ends):
        return
    for o_iso, row in rows.items():
        for f in row["h"].values():
            if f["a"] is not None or pd.Timestamp(f["t"]) > r_ends[-1]:
                continue
            idx = int(r_ends.searchsorted(pd.Timestamp(f["t"]), side="right")) - 1
            if idx >= 0 and r_ends[idx] > pd.Timestamp(o_iso):
                f["a"] = round(math.log(float(ref["close"].iloc[idx]) / row["p0"]) * BP, 4)


def _replay(tf: Timeframe, samples: list[dict], prior_rec: dict | None) -> list[dict]:
    """Replay the live learning rule over pooled backtest forecasts (all pairs), oldest first.

    Each forecast gets the centre, spread and band shape the live forecaster
    would have given it: the rule of ``learning.py``, from the same walk-forward
    prior, fed with the backtest forecasts whose outcome was known at its origin
    (news signal 0)."""
    by_h: dict[str, list[dict]] = {}
    for s in samples:
        by_h.setdefault(s["hkey"], []).append(s)
    out = []
    for hkey, rows in by_h.items():
        rows.sort(key=lambda r: (r["origin"], r["pair"]))
        nu = band_nu(tf.key, int(hkey))
        prior = (prior_rec or {}).get("h", {}).get(hkey)
        known = sorted((r for r in rows if r["a"] is not None), key=lambda r: (r["t"], r["origin"], r["pair"]))
        pos = {id(r): i for i, r in enumerate(known)}
        n = len(known)
        m = np.array([r["m"] for r in known], dtype=float).reshape(n, len(MODEL_KEYS))
        a = np.array([r["a"] for r in known], dtype=float)
        sig = np.array([r["s"] for r in known], dtype=float)
        dft = np.array([r.get("d", 0.0) for r in known], dtype=float)
        used = np.zeros((4, n))              # k, g, c0, c given to each forecast by the replay
        zero = np.zeros(n)
        j, st = 0, None
        for r in rows:
            j0 = j
            while j < n and known[j]["t"] <= r["origin"]:
                j += 1
            if st is None or j != j0:
                st = learn_arrays(prior, m[:j], a[:j], sig[:j], used[0, :j], used[1, :j], used[2, :j], used[3, :j],
                                  zero[:j], tf.half_life, dft[:j])
            r["k"], r["g"], r["c0"] = st.k, st.gain, float(np.dot(st.weights, r["m"]))
            r["c"] = st.gain * r["c0"] + r.get("d", 0.0)
            r["sigma"] = r["s"] * st.k
            r["nu"] = nu
            r["p_up"] = prob_up(r["c"], r["sigma"], nu)
            if id(r) in pos:
                used[:, pos[id(r)]] = (r["k"], r["g"], r["c0"], r["c"])
            out.append(r)
    return out


def _metrics(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["a"] is not None]
    n = len(rows)
    if not n:
        return {"n": 0}
    a = np.array([r["a"] for r in rows])
    c = np.array([r["c"] for r in rows])
    sig = np.array([r["sigma"] for r in rows])
    p = np.array([r["p_up"] for r in rows])
    moving = (np.abs(c) > 1e-9) & (np.abs(a) > 1e-9)
    hits = int(np.sum(np.sign(c[moving]) == np.sign(a[moving])))
    rmse, rmse_rw = float(np.sqrt(np.mean((c - a) ** 2))), float(np.sqrt(np.mean(a ** 2)))
    cover = {}
    for lv in LEVELS:
        zmult = np.array([band_z(r["nu"])[lv] for r in rows])
        cover[lv] = float(np.mean(np.abs(a - c) <= zmult * sig))
    d = np.array([r.get("d", 0.0) for r in rows])
    dt = np.abs(np.array([r.get("dt", 0.0) for r in rows]))
    calls = {}
    for name, sel in (("all", np.abs(d) > 1e-9), ("high", (np.abs(d) > 1e-9) & (dt >= season.T_HIGH))):
        called = moving & sel
        calls[name] = {"n": int(called.sum()), "share": float(np.mean(sel)),
                       "hit": float(np.mean(np.sign(c[called]) == np.sign(a[called]))) if called.any() else None}
    return {"n": n, "hit": hits / moving.sum() if moving.sum() else None, "n_dir": int(moving.sum()), "calls": calls,
            "skill": 1 - rmse / rmse_rw if rmse_rw > 0 else None, "cover": cover,
            "mae_bp": float(np.mean(np.abs(c - a))), "brier": float(np.mean((p - (a > 0)) ** 2))}


def summary(state, tfs: list[Timeframe], pairs: list[str], now: datetime, decimals: dict[str, int],
            overlay: int = 60) -> dict:
    """Backtest accuracy per timeframe and horizon (whole window and recent part, all
    pairs and each pair), a timeline of range coverage and hit rate, and per-pair
    past forecasts for the chart overlay."""
    out = {"at": iso(now), "tf": {}, "pairs": {}}
    version = _version()
    for tf in tfs:
        samples = []
        for code in pairs:
            cache = state.read_cache(_key(tf.key, code), None)
            if not cache or cache.get("v") != version:
                continue
            for o_iso, row in cache["rows"].items():
                for hkey, f in row["h"].items():
                    samples.append({"pair": code, "origin": o_iso, "hkey": hkey, "p0": row["p0"], **f})
        if not samples:
            continue
        prior = next((r for r in reversed(state.ledger.records) if r["type"] == "prior" and r["tf"] == tf.key), None)
        rows = _replay(tf, samples, prior)
        recent_from = iso(now - RECENT[tf.key])
        by_h, by_pair, timeline = {}, {}, {}
        freq, win, roll_label = ROLL[tf.key]
        for h in tf.horizons:
            hr = sorted((r for r in rows if r["hkey"] == str(h)), key=lambda r: r["t"])
            by_h[str(h)] = {"all": _metrics(hr), "recent": _metrics([r for r in hr if r["t"] >= recent_from])}
            for code in pairs:
                pr = [r for r in hr if r["pair"] == code]
                if pr:
                    by_pair.setdefault(code, {})[str(h)] = {"all": _metrics(pr),
                                                            "recent": _metrics([r for r in pr if r["t"] >= recent_from])}
            # coverage and hit rate over a trailing window, all pairs together
            df = pd.DataFrame([{"t": r["t"], "in80": float(abs(r["a"] - r["c"]) <= band_z(r["nu"])["80"] * r["sigma"]),
                                "hit": float(np.sign(r["c"]) == np.sign(r["a"]))
                                if abs(r["c"]) > 1e-9 and abs(r["a"]) > 1e-9 else np.nan}
                               for r in hr if r["a"] is not None])
            if len(df):
                g = df.set_index(pd.to_datetime(df["t"])).resample(freq, label="right")
                agg = pd.DataFrame({"in80": g["in80"].sum(), "hit": g["hit"].sum(), "n": g.size(), "n_dir": g["hit"].count()})
                agg = agg[agg["n"] > 0].rolling(win, min_periods=1).sum()
                timeline[str(h)] = [[iso(ts.to_pydatetime()), round(x.in80 / x.n, 4),
                                     round(x.hit / x.n_dir, 4) if x.n_dir else None, int(x.n)]
                                    for ts, x in agg.iterrows()]
        out["tf"][tf.key] = {"window_days": WINDOW[tf.key].days or 1, "recent_hours": int(RECENT[tf.key].total_seconds() // 3600),
                             "origins": len({(r["pair"], r["origin"]) for r in rows}), "by_h": by_h, "by_pair": by_pair,
                             "timeline": timeline, "roll": roll_label}
        # chart overlay: the latest scored backtest forecasts per pair and horizon
        for code in pairs:
            dec = decimals.get(code, 3) + 1
            per_h = {}
            for h in tf.horizons:
                sel = sorted((r for r in rows if r["pair"] == code and r["hkey"] == str(h) and r["a"] is not None),
                             key=lambda r: r["t"])[-overlay:]
                items = []
                for r in sel:
                    z = band_z(r["nu"])
                    px = lambda bp: round(r["p0"] * math.exp(bp / BP), dec)  # noqa: E731
                    items.append({"t": r["t"], "origin": r["origin"], "c": px(r["c"]), "actual": px(r["a"]),
                                  **{f"{s}{lv}": px(r["c"] + sg * z[lv] * r["sigma"]) for lv in LEVELS
                                     for s, sg in (("lo", -1), ("hi", 1))},
                                  "in80": abs(r["a"] - r["c"]) <= z["80"] * r["sigma"]})
                per_h[str(h)] = items
            out["pairs"].setdefault(code, {})[tf.key] = per_h
    return out


__all__ = ["update", "summary", "parse_iso"]
