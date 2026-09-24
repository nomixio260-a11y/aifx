"""Online learning from the forecaster's own scored predictions.

Everything here is a pure function of (walk-forward prior, scored outcomes),
so the state used by any past prediction can be recomputed by an auditor
from the ledger alone. Four things are learned per timeframe and horizon,
pooled across pairs:

* model weights: inverse of each model's recency-weighted mean squared
  standardized error, blended with the walk-forward prior;
* interval scale k: stretches or shrinks the predicted spread so the 80 %
  range actually covers ~80 % of outcomes;
* gain g: how much of the blended model signal to trust. It is the
  regression slope of outcomes on forecasts (both in units of the forecast's
  spread), shrunk toward 0. Without demonstrated skill the forecast stays
  near "no change"; a negative gain means the signal has been working in
  reverse (e.g. hourly trends tending to fade);
* news coefficient beta: how much the news signal should tilt the forecast,
  in units of the forecast's standard deviation. It starts from a small prior
  and only moves when live results support it; news can't be backtested
  honestly because historical headlines were not collected point-in-time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .engine import BAND_Z, MODEL_KEYS

PRIOR_N_MSE = 40.0
PRIOR_N_K = 30.0
BETA_PRIOR = 0.05
BETA_LAMBDA = 20.0
K_BOUNDS = (0.6, 2.5)
# Gain prior: centred on "no skill" with a spread of about 0.14 (lambda = 1 / 0.14^2).
# Pooled samples from 7 correlated pairs carry far less information than their
# count suggests, so they are down-weighted to an effective sample share.
# Replaying this rule over 2002-2026 (daily) and two years of hourly bars, a
# looser prior (lambda 11) made 10-20 day forecasts slightly worse than "no
# change"; lambda 50 was better on the tuning period and held up on the test
# period (research/report.md).
GAIN_LAMBDA = 50.0
GAIN_ESS = 0.35
GAIN_BOUNDS = (-0.5, 1.0)
BETA_BOUNDS = (-0.5, 0.5)


@dataclass
class HorizonState:
    weights: list[float]
    k: float
    beta: float
    gain: float
    n_live: int
    n_eff: float
    mse_z: list[float]

    def as_dict(self) -> dict:
        return {"w": self.weights, "k": self.k, "b": self.beta, "g": self.gain, "n": self.n_live,
                "n_eff": round(self.n_eff, 3), "mse_z": self.mse_z}


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, q * cum[-1]))
    return float(v[min(idx, len(v) - 1)])


def learn_horizon(prior: dict | None, samples: list[dict], half_life: float) -> HorizonState:
    """``samples``: scored forecasts for one horizon, sorted oldest first.

    Each sample has m (model bp list), a (actual bp), s (raw sigma bp), k (scale
    used), g (gain used), c0 (model blend), c (final centre) and x (news signal).
    """
    col = {key: np.array([s[key] for s in samples], dtype=float) for key in ("a", "s", "k", "g", "c0", "c", "x")}
    m = np.array([s["m"] for s in samples], dtype=float).reshape(len(samples), len(MODEL_KEYS))
    return learn_arrays(prior, m, half_life=half_life, **col)


def learn_arrays(prior: dict | None, m: np.ndarray, a: np.ndarray, s: np.ndarray, k: np.ndarray, g: np.ndarray,
                 c0: np.ndarray, c: np.ndarray, x: np.ndarray, half_life: float) -> HorizonState:
    """The same rule on columns (one row per sample, oldest first); used to replay it quickly."""
    n_models = len(MODEL_KEYS)
    prior_mse = np.array(prior["mse_z"]) if prior else np.ones(n_models)
    prior_k = float(prior["k"]) if prior else 1.0
    suu = float(prior.get("suu", 0.0)) if prior else 0.0
    suv = float(prior.get("suv", 0.0)) if prior else 0.0
    n = len(a)
    if n:
        ranks = np.arange(n)
        w = 0.5 ** ((n - 1 - ranks) / half_life)
        z2 = ((m - a[:, None]) / s[:, None]) ** 2
        w_sum = float(w.sum())
        mse = (PRIOR_N_MSE * prior_mse + (w[:, None] * z2).sum(axis=0)) / (PRIOR_N_MSE + w_sum)
        ze = np.abs(c - a) / s
        k_hat = _weighted_quantile(ze, w, 0.8) / BAND_Z["80"]
        k_new = (PRIOR_N_K * prior_k + w_sum * k_hat) / (PRIOR_N_K + w_sum)
        u, v = c0 / s, a / s
        suu += float(np.sum(w * u * u))
        suv += float(np.sum(w * u * v))
        resid = (a - g * c0) / (s * k)
        beta = (BETA_LAMBDA * BETA_PRIOR + float(np.sum(w * resid * x))) / (BETA_LAMBDA + float(np.sum(w * x * x)))
    else:
        w_sum = 0.0
        mse = prior_mse
        k_new = prior_k
        beta = BETA_PRIOR
    gain = GAIN_ESS * suv / (GAIN_LAMBDA + GAIN_ESS * suu)
    inv = 1.0 / np.maximum(mse, 1e-9)
    weights = inv / inv.sum()
    return HorizonState(
        weights=[round(float(v), 6) for v in weights],
        k=round(float(np.clip(k_new, *K_BOUNDS)), 6),
        beta=round(float(np.clip(beta, *BETA_BOUNDS)), 6),
        gain=round(float(np.clip(gain, *GAIN_BOUNDS)), 6),
        n_live=n,
        n_eff=w_sum,
        mse_z=[round(float(v), 6) for v in mse],
    )


def samples_from_ledger(predictions: dict[int, dict], outcomes: list[dict], tf: str,
                        upto_seq: int | None = None) -> dict[int, list[dict]]:
    """Join outcome items with their predictions, for one timeframe, per horizon.

    Only outcome records with seq <= ``upto_seq`` are used, which is what makes
    the learning state at any past prediction reproducible.
    """
    by_h: dict[int, list[tuple]] = {}
    for rec in outcomes:
        if upto_seq is not None and rec["seq"] > upto_seq:
            break
        for pred_seq, h, actual, _bar_end in rec["items"]:
            p = predictions.get(pred_seq)
            if p is None or p["tf"] != tf:
                continue
            fc = next((f for f in p["fc"] if f["h"] == h), None)
            if fc is None:
                continue
            a = float(np.log(actual / p["p0"]) * 1e4)
            by_h.setdefault(h, []).append((fc["t"], pred_seq, {
                "m": fc["m"], "a": a, "s": fc["s"], "k": fc["k"], "g": fc["g"], "c0": fc["c0"],
                "c": fc["c"], "x": p["news"]["x"],
            }))
    return {h: [s for _, _, s in sorted(rows, key=lambda r: (r[0], r[1]))] for h, rows in by_h.items()}


def learn(prior_rec: dict | None, samples: dict[int, list[dict]], horizons, half_life: float) -> dict[int, HorizonState]:
    out = {}
    for h in horizons:
        prior = prior_rec["h"].get(str(h)) if prior_rec else None
        out[h] = learn_horizon(prior, samples.get(h, []), half_life)
    return out
