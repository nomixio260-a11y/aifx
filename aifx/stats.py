"""Statistics for judging forecasts without flattering them.

* Direction hit rates come with a Wilson 95 % interval and a test against a
  coin flip, so 60 % on 10 forecasts is not mistaken for skill.
* Point accuracy is compared with the random walk (no change) on exactly the
  same forecasts, with a Diebold-Mariano test using a Newey-West variance
  (overlapping multi-step forecasts are autocorrelated).
* Probabilities are scored with the Brier score against always saying 50 %.
"""

from __future__ import annotations

import math

import numpy as np


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float | None:
    """Two-sided p-value for k successes in n trials."""
    if n <= 0:
        return None
    if n <= 400:
        probs = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
        obs = probs[k]
        return float(min(1.0, sum(q for q in probs if q <= obs * (1 + 1e-9))))
    mean, sd = n * p, math.sqrt(n * p * (1 - p))
    z = (abs(k - mean) - 0.5) / sd
    return float(min(1.0, 2 * norm_sf(max(z, 0.0))))


def diebold_mariano(loss_model: np.ndarray, loss_base: np.ndarray, lag: int = 0) -> tuple[float | None, float | None]:
    """DM statistic and two-sided p-value; negative statistic = model better."""
    d = np.asarray(loss_model, float) - np.asarray(loss_base, float)
    n = len(d)
    if n < 8:
        return None, None
    dm = d - d.mean()
    var = float(dm @ dm) / n
    for L in range(1, min(lag, n - 1) + 1):
        cov = float(dm[L:] @ dm[:-L]) / n
        var += 2 * (1 - L / (lag + 1)) * cov
    if var <= 0:
        return None, None
    stat = float(d.mean() / math.sqrt(var / n))
    return stat, float(2 * norm_sf(abs(stat)))


def scores(c: np.ndarray, a: np.ndarray, sigma: np.ndarray, p_up: np.ndarray, lag: int = 0,
           band: dict[str, np.ndarray] | None = None) -> dict:
    """Headline metrics for forecasts c (bp) against outcomes a (bp).

    ``band``: half-width of each range in units of sigma, per forecast
    (default: normal-distribution bands).
    """
    n = len(a)
    if n == 0:
        return {"n": 0}
    moving = (np.abs(c) > 1e-9) & (np.abs(a) > 1e-9)
    hits = int(np.sum(np.sign(c[moving]) == np.sign(a[moving])))
    n_dir = int(moving.sum())
    lo, hi = wilson(hits, n_dir)
    err = c - a
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rmse_rw = float(np.sqrt(np.mean(a ** 2)))
    dm, dm_p = diebold_mariano(err ** 2, a ** 2, lag)
    up = (a > 0).astype(float)
    brier = float(np.mean((p_up - up) ** 2))
    cover = {}
    for name, z in (("50", 0.6745), ("80", 1.2816), ("95", 1.96)):
        if band is not None:
            z = band[name]
        inside = np.abs(err) <= z * sigma
        k = int(inside.sum())
        cl, ch = wilson(k, n)
        cover[name] = {"rate": k / n, "lo": cl, "hi": ch}
    return {
        "n": n,
        "direction": {"n": n_dir, "hits": hits, "rate": hits / n_dir if n_dir else None,
                      "lo": lo, "hi": hi, "p": binom_two_sided(hits, n_dir)},
        "rmse_bp": rmse,
        "rmse_rw_bp": rmse_rw,
        "skill": (1 - rmse / rmse_rw) if rmse_rw > 0 else None,
        "dm": dm,
        "dm_p": dm_p,
        "mae_bp": float(np.mean(np.abs(err))),
        "brier": brier,
        "bss": 1 - brier / 0.25,
        "coverage": cover,
    }
