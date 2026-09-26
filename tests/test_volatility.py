from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from aifx.api import research_summary
from aifx.engine import BAND_Z, TIMEFRAMES, band_z, horizon_sigma, prob_up, sigma_steps
from aifx.stats import scores
from aifx.timeutil import london_day_end
from aifx.volatility import (daily_variance_inputs, range_variance, realized_daily_variance, scale_proxy)

UTC = timezone.utc


def test_t_bands_keep_the_80_band_and_widen_the_tails():
    assert band_z(None) == BAND_Z
    for nu in (5, 10, 30):
        z = band_z(nu)
        assert z["80"] == pytest.approx(BAND_Z["80"])
        assert z["50"] < BAND_Z["50"] and z["95"] > BAND_Z["95"]
    assert band_z(5)["95"] > band_z(30)["95"]          # fewer degrees of freedom, fatter tails


def test_probability_of_a_rise_is_symmetric_and_monotone():
    for nu in (None, 5, 30):
        assert prob_up(0.0, 10.0, nu) == pytest.approx(0.5, abs=1e-9)
        assert prob_up(3.0, 10.0, nu) + prob_up(-3.0, 10.0, nu) == pytest.approx(1.0, abs=1e-9)
        assert prob_up(1.0, 10.0, nu) < prob_up(5.0, 10.0, nu) < 1
    # the 80 % band edges carry the same probability under both shapes
    assert prob_up(-BAND_Z["80"] * 10, 10.0, 5) == pytest.approx(0.1, abs=1e-4)
    assert prob_up(1.0, 0.0, 5) == 0.5


def test_range_variance_and_scaling():
    bars = pd.DataFrame({"open": [1, 1, 1, 1], "high": [1.0, 1.02, 1.0, 1.01], "low": [1.0, 1.0, 1.0, 0.99],
                         "close": [1, 1, 1, 1]}, dtype=float)
    rv = range_variance(bars)
    assert rv[0] == pytest.approx(np.log(1.02) ** 2 / (4 * np.log(2)))
    assert np.isnan(rv[1])                              # no range: unusable
    r = np.full(200, 0.01)
    assert scale_proxy(np.full(200, np.nan), r, 100) is None
    alt = np.full(200, 4e-4)
    alt[-1] = np.nan
    sq = scale_proxy(alt, r, 100)
    assert sq[0] == pytest.approx(1e-4) and sq[-1] == pytest.approx(1e-4)


def _hourly(start: str, n: int, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    idx = idx[[not (t.dayofweek == 5 or (t.dayofweek == 4 and t.hour >= 21) or (t.dayofweek == 6 and t.hour < 21))
               for t in idx]]
    close = 150 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.001, len(idx))))
    return pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close}, index=idx)


def test_realized_variance_groups_by_london_day_and_stops_at_the_cutoff():
    h = _hourly("2026-09-01", 24 * 20)
    until = datetime(2026, 9, 15, 23, tzinfo=UTC)          # London midnight in summer time
    rv = realized_daily_variance(h, until)
    assert rv.index.max() == pd.Timestamp("2026-09-15")
    assert all(d.dayofweek < 5 for d in rv.index)           # Sunday evening belongs to Monday
    r = np.diff(np.log(h["close"].to_numpy()))
    day = (h.index[1:] + pd.Timedelta(minutes=59)).tz_convert("Europe/London").normalize().tz_localize(None)
    sel = day == pd.Timestamp("2026-09-10")
    assert rv[pd.Timestamp("2026-09-10")] == pytest.approx(float(np.sum(r[sel] ** 2)))


def test_daily_sigma_uses_only_hourly_bars_up_to_the_origin():
    h = _hourly("2025-06-01", 24 * 500, seed=2)
    days = pd.bdate_range("2023-01-02", "2026-09-15")
    close = h["close"].reindex(pd.DatetimeIndex([london_day_end(d.date()) - pd.Timedelta(hours=1) for d in days]),
                               method="ffill").to_numpy()
    close = np.where(np.isnan(close), 150.0, close) * np.exp(np.random.default_rng(3).normal(0, 1e-4, len(days)))
    daily = pd.DataFrame({"open": close, "high": close, "low": close, "close": close}, index=days)
    origin = london_day_end(days[-1].date())
    tf = TIMEFRAMES["1d"]
    with_future = sigma_steps(tf, daily.iloc[-1000:], origin, 20, hourly=h)
    upto = h[h.index + pd.Timedelta(hours=1) <= pd.Timestamp(origin)]
    assert np.allclose(with_future, sigma_steps(tf, daily.iloc[-1000:], origin, 20, hourly=upto))
    without = sigma_steps(tf, daily.iloc[-1000:], origin, 20)
    assert not np.allclose(with_future, without)
    sq, lam = daily_variance_inputs(daily.iloc[-1000:], h, origin)
    assert sq is not None and lam < 0.94
    assert daily_variance_inputs(daily.iloc[-1000:], None, origin)[0] is None


def test_hourly_sigma_follows_the_high_low_range():
    h = _hourly("2026-03-02", 24 * 180, seed=4)
    origin = h.index[-1].to_pydatetime() + pd.Timedelta(hours=1)
    base = horizon_sigma(sigma_steps(TIMEFRAMES["1h"], h, origin, 24), (1, 4, 24))
    wide = h.copy()
    wide.iloc[-30:, wide.columns.get_loc("high")] = wide["close"].iloc[-30:] * 1.01
    wide.iloc[-30:, wide.columns.get_loc("low")] = wide["close"].iloc[-30:] * 0.99
    assert horizon_sigma(sigma_steps(TIMEFRAMES["1h"], wide, origin, 24), (1, 4, 24))[0] > base[0]


def test_coverage_uses_each_forecasts_band_shape():
    a = np.array([0.7, -0.7, 2.1, 0.1])
    c = np.zeros(4)
    sigma = np.ones(4)
    p = np.full(4, 0.5)
    normal = scores(c, a, sigma, p)
    t5 = band_z(5)
    fat = scores(c, a, sigma, p, band={k: np.full(4, v) for k, v in t5.items()})
    assert normal["coverage"]["95"]["rate"] == 0.75 and fat["coverage"]["95"]["rate"] == 1.0
    assert normal["coverage"]["50"]["rate"] == 0.25 and fat["coverage"]["50"]["rate"] == 0.25


def test_research_summary_is_optional(tmp_path):
    assert research_summary(tmp_path / "missing.json") is None
    s = research_summary()
    if s is not None:
        assert {"1d", "1h"} <= set(s["tf"]) <= {"15m", "1d", "1h"}
        assert all(set(x["h"]) == set(s["tf"][tf]["ranges"]) for tf in s["tf"] for x in s["tf"][tf]["direction"])


def test_hourly_ranges_are_measured_over_the_whole_profile_window(tmp_path, monkeypatch):
    """The time-of-day profile needs about a year of hourly bars, more than the models are fitted on."""
    from aifx import engine, forecaster

    from .conftest import run_session
    seen: dict[str, list[int]] = {}
    real = forecaster.sigma_steps

    def spy(tf, bars, *a, **k):
        seen.setdefault(tf.key, []).append(len(bars))
        return real(tf, bars, *a, **k)

    monkeypatch.setattr(forecaster, "sigma_steps", spy)
    run_session(tmp_path / "state", cycles=1)
    assert min(seen["1h"]) > engine.TIMEFRAMES["1h"].fit_bars
    assert max(seen["1h"]) <= engine.HOURLY_PROFILE_WINDOW + 500
    assert max(seen["1d"]) <= engine.TIMEFRAMES["1d"].fit_bars
