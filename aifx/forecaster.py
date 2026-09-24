"""Build one prediction record from inputs that existed at its origin.

The live pipeline and the auditor both call :func:`make_prediction`, so an
auditor who reconstructs the inputs from the ledger (bars committed before the
prediction, headlines stored before its origin, the learning state at the
time) must get the same numbers back. Any use of information from after the
origin would show up as a mismatch.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import news as newsmod
from .data import Pair
from .engine import (BAND_Z, BP, MODEL_KEYS, Timeframe, combine, horizon_sigma, model_paths,
                     sigma_steps, step_ends, target_times)
from .learning import HorizonState
from .timeutil import iso

MAX_ORIGIN_AGE = {"1h": timedelta(hours=2), "1d": timedelta(hours=36)}
_MODEL_FILES = ("models.py", "engine.py", "volatility.py", "learning.py", "news.py", "forecaster.py")


def model_version() -> str:
    h = hashlib.sha256()
    here = Path(__file__).parent
    for name in _MODEL_FILES:
        h.update((here / name).read_bytes())
    return h.hexdigest()[:12]


def origin_of(tf: Timeframe, bars: pd.DataFrame) -> datetime:
    if tf.key == "1h":
        return bars.index[-1].to_pydatetime() + timedelta(hours=1)
    from .timeutil import london_day_end
    return london_day_end(bars.index[-1].date())


def origin_price(hourly: pd.DataFrame, origin: datetime) -> tuple[float, str] | None:
    """Close of the latest hourly bar ending at or before ``origin``."""
    ends = hourly.index + pd.Timedelta(hours=1)
    idx = int(ends.searchsorted(pd.Timestamp(origin), side="right")) - 1
    if idx < 0:
        return None
    return float(hourly["close"].iloc[idx]), iso(ends[idx].to_pydatetime())


def make_prediction(tf: Timeframe, pair: Pair, bars: pd.DataFrame, hourly: pd.DataFrame, origin: datetime,
                    news_items: list[dict], events: list[dict], prior_seq: int, learn_seq: int,
                    state: dict[int, HorizonState], version: str) -> tuple[dict, dict]:
    """Returns (ledger record without seq/hash/at, chart detail)."""
    y = np.log(bars["close"].to_numpy()[-tf.fit_bars:])
    steps = max(tf.steps, max(tf.horizons))
    paths, analog_idx = model_paths(y, steps)
    ends = step_ends(tf, origin, steps)
    evs = newsmod.events_between(events, (pair.base, pair.quote), origin, ends[-1], origin)
    var = sigma_steps(tf, bars.iloc[-tf.fit_bars:], origin, steps, evs)
    press = newsmod.pressures(news_items, origin)
    x = newsmod.pair_signal(press, pair.base, pair.quote)
    p0, p0_bar = origin_price(hourly, origin)
    targets = target_times(tf, origin)
    sig_h = horizon_sigma(var, tf.horizons)
    fc = []
    for j, h in enumerate(tf.horizons):
        st = state[h]
        m = {k: float(paths[k][h - 1]) for k in MODEL_KEYS}
        comb = combine(m, st.weights, sig_h[j], st.k, x, st.beta, st.gain)
        n_ev = sum(1 for e in evs if origin < e["time"] <= targets[j])
        fc.append({
            "h": h,
            "t": iso(targets[j]),
            "m": [round(m[k], 4) for k in MODEL_KEYS],
            "w": st.weights,
            "s": round(sig_h[j], 4),
            "k": st.k,
            "b": st.beta,
            "g": st.gain,
            "c0": round(comb["c0"], 4),
            "c": round(comb["c"], 4),
            "p": round(comb["p_up"], 4),
            "ev": n_ev,
        })
    record = {
        "type": "prediction",
        "pair": pair.code,
        "tf": tf.key,
        "origin": iso(origin),
        "p0": p0,
        "p0_bar": p0_bar,
        "prior": prior_seq,
        "learn": learn_seq,
        "news": {"x": x, "n": sum(v["n"] for v in press.values()), "cut": iso(origin)},
        "models": MODEL_KEYS,
        "v": version,
        "fc": fc,
    }
    chart = chart_detail(tf, pair, bars, origin, p0, paths, var, state, x, ends, evs, analog_idx, press)
    return record, chart


def chart_detail(tf, pair, bars, origin, p0, paths, var, state, x, ends, evs, analog_idx, press) -> dict:
    """Per-step paths for drawing (not part of the ledger)."""
    steps = len(ends)
    hs = sorted(state)
    cum = np.sqrt(np.cumsum(var))
    rows = []
    model_steps = {k: [] for k in MODEL_KEYS}
    for i in range(steps):
        h_ref = next((h for h in hs if h >= i + 1), hs[-1])
        st = state[h_ref]
        m = {k: float(paths[k][i]) for k in MODEL_KEYS}
        comb = combine(m, st.weights, float(cum[i]), st.k, x, st.beta, st.gain)
        row = {"t": iso(ends[i]), "c": p0 * float(np.exp(comb["c"] / BP)), "p": comb["p_up"]}
        for name, z in BAND_Z.items():
            row["lo" + name] = p0 * float(np.exp((comb["c"] - z * comb["sigma"]) / BP))
            row["hi" + name] = p0 * float(np.exp((comb["c"] + z * comb["sigma"]) / BP))
        rows.append(row)
        for k in MODEL_KEYS:
            model_steps[k].append(p0 * float(np.exp(m[k] / BP)))
    analogs = []
    for i in analog_idx:
        pos = len(bars) - tf.fit_bars + i if len(bars) > tf.fit_bars else i
        if 0 <= pos < len(bars):
            ts = bars.index[pos]
            analogs.append(ts.strftime("%Y-%m-%d %H:%M") if tf.key == "1h" else ts.strftime("%Y-%m-%d"))
    return {
        "origin": iso(origin),
        "p0": p0,
        "steps": rows,
        "models": model_steps,
        "events": [{"time": iso(e["time"]), "impact": e["impact"], "cur": e["cur"], "title": e["title"]} for e in evs],
        "news": {"x": x, "base": press[pair.base], "quote": press[pair.quote]},
        "analogs": analogs,
    }
