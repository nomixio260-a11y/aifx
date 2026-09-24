"""The live track record: every scored prediction, joined with its outcome."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .engine import BAND_Z, BP, MODEL_KEYS, TIMEFRAMES, band_z
from .stats import scores


def join(predictions: list[dict], outcomes: list[dict]) -> tuple[list[dict], set]:
    """Scored samples (oldest target first) and the set of (seq, h) already scored or void."""
    by_seq = {p["seq"]: p for p in predictions}
    rows = []
    done = set()
    for o in outcomes:
        for pseq, h, actual, bar_end in o["items"]:
            done.add((pseq, h))
            p = by_seq.get(pseq)
            if p is None or actual is None:
                continue
            f = next(x for x in p["fc"] if x["h"] == h)
            a = math.log(actual / p["p0"]) * BP
            rows.append({
                "seq": pseq, "pair": p["pair"], "tf": p["tf"], "h": h, "issued": p["at"], "origin": p["origin"],
                "target": f["t"], "p0": p["p0"], "c": f["c"], "c0": f["c0"], "g": f["g"],
                "sigma": f["s"] * f["k"], "p": f["p"], "z": band_z(f.get("nu")),
                "m": f["m"], "x": p["news"]["x"], "actual": actual, "a": a, "bar_end": bar_end, "scored": o["at"],
            })
    rows.sort(key=lambda r: (r["target"], r["seq"], r["h"]))
    return rows, done


def _arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def summary(rows: list[dict], tf: str, h: int) -> dict:
    sel = [r for r in rows if r["tf"] == tf and r["h"] == h]
    if not sel:
        return {"n": 0}
    lag = max(0, h - 1)
    z = {name: np.array([r["z"][name] for r in sel]) for name in BAND_Z}
    out = scores(_arr(sel, "c"), _arr(sel, "a"), _arr(sel, "sigma"), _arr(sel, "p"), lag, z)
    a = _arr(sel, "a")
    m = np.array([r["m"] for r in sel])
    rmse_rw = float(np.sqrt(np.mean(a ** 2)))
    out["models"] = {
        k: {"rmse_bp": float(np.sqrt(np.mean((m[:, i] - a) ** 2))),
            "skill": (1 - float(np.sqrt(np.mean((m[:, i] - a) ** 2))) / rmse_rw) if rmse_rw > 0 else None}
        for i, k in enumerate(MODEL_KEYS)
    }
    no_news = np.array([r["g"] * r["c0"] for r in sel])
    out["news_effect"] = {
        "rmse_with_bp": out["rmse_bp"],
        "rmse_without_bp": float(np.sqrt(np.mean((no_news - a) ** 2))),
        "active": int(np.sum(np.abs(_arr(sel, "x")) > 1e-9)),
    }
    return out


def timeline(rows: list[dict], tf: str, h: int, max_points: int = 160) -> list[list]:
    sel = [r for r in rows if r["tf"] == tf and r["h"] == h]
    out = []
    hits = n_dir = 0
    se = se_rw = 0.0
    inside = 0
    step = max(1, len(sel) // max_points)
    for i, r in enumerate(sel, start=1):
        if abs(r["c"]) > 1e-9 and abs(r["a"]) > 1e-9:
            n_dir += 1
            hits += int((r["c"] > 0) == (r["a"] > 0))
        se += (r["c"] - r["a"]) ** 2
        se_rw += r["a"] ** 2
        inside += int(abs(r["c"] - r["a"]) <= BAND_Z["80"] * r["sigma"])
        if i % step == 0 or i == len(sel):
            out.append([r["target"], round(hits / n_dir, 4) if n_dir else None,
                        round(1 - math.sqrt(se / se_rw), 4) if se_rw > 0 else None, round(inside / i, 4), i])
    return out


def calibration(rows: list[dict], tf: str, bins: int = 10) -> list[dict]:
    sel = [r for r in rows if r["tf"] == tf]
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        s = [r for r in sel if lo <= r["p"] < hi or (b == bins - 1 and r["p"] == 1.0)]
        if s:
            out.append({"lo": lo, "hi": hi, "n": len(s), "p": float(np.mean([r["p"] for r in s])),
                        "freq": float(np.mean([r["a"] > 0 for r in s]))})
    return out


def build(predictions: list[dict], outcomes: list[dict], pip_of: dict[str, float]) -> dict:
    rows, done = join(predictions, outcomes)
    overall = {tf: {str(h): summary(rows, tf, h) for h in TIMEFRAMES[tf].horizons} for tf in TIMEFRAMES}
    by_pair: dict = defaultdict(dict)
    for pair in sorted({r["pair"] for r in rows}):
        pr = [r for r in rows if r["pair"] == pair]
        for tf in TIMEFRAMES:
            by_pair[pair][tf] = {}
            for h in TIMEFRAMES[tf].horizons:
                s = summary(pr, tf, h)
                by_pair[pair][tf][str(h)] = {k: s.get(k) for k in ("n", "direction", "skill", "rmse_bp", "coverage")}
    recent = [_row_view(r, pip_of) for r in reversed(rows[-150:])]
    pending = []
    for p in predictions:
        for f in p["fc"]:
            if (p["seq"], f["h"]) not in done:
                pending.append({"seq": p["seq"], "pair": p["pair"], "tf": p["tf"], "h": f["h"], "issued": p["at"],
                                "target": f["t"], "p0": p["p0"], "price": p["p0"] * math.exp(f["c"] / BP),
                                "p_up": f["p"]})
    pending.sort(key=lambda r: r["target"])
    return {
        "overall": overall,
        "by_pair": dict(by_pair),
        "recent": recent,
        "pending": pending[:60],
        "pending_count": len(pending),
        "timeline": {tf: {str(h): timeline(rows, tf, h) for h in TIMEFRAMES[tf].horizons} for tf in TIMEFRAMES},
        "calibration": {tf: calibration(rows, tf) for tf in TIMEFRAMES},
        "scored": len(rows),
    }


def _row_view(r: dict, pip_of: dict[str, float]) -> dict:
    pred = r["p0"] * math.exp(r["c"] / BP)
    z80 = BAND_Z["80"]
    moved = abs(r["a"]) > 1e-9 and abs(r["c"]) > 1e-9
    return {
        "seq": r["seq"], "pair": r["pair"], "tf": r["tf"], "h": r["h"], "issued": r["issued"], "target": r["target"],
        "p0": r["p0"], "price": pred, "actual": r["actual"], "p_up": r["p"],
        "lo80": r["p0"] * math.exp((r["c"] - z80 * r["sigma"]) / BP),
        "hi80": r["p0"] * math.exp((r["c"] + z80 * r["sigma"]) / BP),
        "err_pips": (r["actual"] - pred) / pip_of.get(r["pair"], 0.01),
        "hit": ((r["c"] > 0) == (r["a"] > 0)) if moved else None,
        "inside80": abs(r["c"] - r["a"]) <= z80 * r["sigma"],
    }
