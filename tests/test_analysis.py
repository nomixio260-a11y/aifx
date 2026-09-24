import math

import numpy as np
import pandas as pd
import pytest

from aifx import analysis
from aifx.api import _coarse_path, distribution
from aifx.data import PAIRS
from aifx.engine import BAND_Z, TIMEFRAMES, prob_up


def test_currency_strength_recovers_the_currency_moves():
    true = {"USD": 0.6, "JPY": -0.5, "EUR": 0.1, "GBP": -0.3, "AUD": 0.1}
    changes = {code: true[p.base] - true[p.quote] for code, p in PAIRS.items()}
    got = analysis.strength(changes)
    for cur, v in true.items():
        assert got[cur] == pytest.approx(v, abs=1e-9)
    assert sum(got.values()) == pytest.approx(0, abs=1e-9)
    assert analysis.strength({"USDJPY": 0.1}) == {}


def test_strength_history_starts_at_zero_and_follows_the_pairs():
    idx = pd.date_range("2026-09-01", periods=30, freq="h", tz="UTC")
    usd = np.linspace(0, 0.01, 30)          # the dollar rises against everything
    hourly = {}
    for code, p in PAIRS.items():
        sign = 1 if p.base == "USD" else -1 if p.quote == "USD" else 0
        hourly[code] = pd.DataFrame({"close": 100 * np.exp(sign * usd)}, index=idx)
    hist = analysis.strength_history(hourly, bars=24)
    assert len(hist["t"]) == 25
    assert all(abs(v[0]) < 1e-9 for v in hist["s"].values())
    assert hist["s"]["USD"][-1] > 0 and hist["s"]["JPY"][-1] < 0


def test_distribution_matches_the_forecast_bands_and_probability():
    pair = PAIRS["USDJPY"]
    p0, c, sig, nu = 150.0, 3.0, 40.0, 5
    d = distribution(p0, c, sig, nu, pair)
    assert d["q"]["0.1"] == pytest.approx(p0 * math.exp((c - BAND_Z["80"] * sig) / 1e4), abs=1e-3)
    assert d["q"]["0.9"] == pytest.approx(p0 * math.exp((c + BAND_Z["80"] * sig) / 1e4), abs=1e-3)
    probs = [lv["p_above"] for lv in d["levels"]]
    assert probs == sorted(probs)                     # levels listed high to low
    at_p0 = [x for x in d["curve"] if abs(x[0] - p0) < 0.05]
    assert at_p0 and abs(at_p0[0][2] - prob_up(c, sig, nu)) < 0.02
    assert max(x[1] for x in d["curve"]) == 1.0


def test_coarse_path_passes_through_the_recorded_horizons():
    rec = {"origin": "2026-09-24T10:00:00Z", "p0": 150.0}
    hz = []
    for h, w in ((1, 0.1), (4, 0.3), (24, 1.0)):
        hz.append({"h": h, "price": 150.0 + 0.01 * h, "p_up": 0.5 + 0.001 * h,
                   **{f"{s}{lv}": 150.0 + sg * w * f for lv, f in (("50", 0.5), ("80", 1.0), ("95", 1.5))
                      for s, sg in (("lo", -1), ("hi", 1))}})
    path = _coarse_path(rec, hz, "1h", 4)
    assert path["coarse"] and len(path["steps"]) == TIMEFRAMES["1h"].steps
    for h in hz:
        st = path["steps"][h["h"] - 1]
        assert st["c"] == pytest.approx(h["price"]) and st["hi80"] == pytest.approx(h["hi80"])
    assert path["steps"][1]["hi80"] < path["steps"][2]["hi80"] < hz[1]["hi80"] + 1e-9
