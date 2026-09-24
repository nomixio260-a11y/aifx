import numpy as np
import pytest

from aifx.data import synthetic_prices
from aifx.models import (AutoRegressive, DampedHolt, Drift, PatternMatch, RandomWalk, RidgeDirect,
                         default_models, technical_features)

H = 20


@pytest.fixture(scope="module")
def y():
    return np.log(synthetic_prices(n=900, seed=3)["close"].to_numpy())


@pytest.mark.parametrize("model", default_models(), ids=lambda m: m.key)
def test_forecast_shape_and_finite(model, y):
    f = model.forecast(y, H)
    assert f.shape == (H,)
    assert np.all(np.isfinite(f))
    # A daily FX model should stay within a plausible distance of the last price.
    assert np.max(np.abs(f - y[-1])) < 0.2


def test_random_walk_repeats_last_value(y):
    assert np.allclose(RandomWalk().forecast(y, 5), y[-1])


def test_trend_followers_pick_up_a_clean_trend():
    y = np.log(100) + 0.002 * np.arange(600)
    assert Drift().forecast(y, 10)[-1] == pytest.approx(y[-1] + 0.02, rel=1e-9)
    holt = DampedHolt().forecast(y, 10)
    assert holt[-1] > y[-1] + 0.01


def test_ar_recovers_mean_reversion():
    rng = np.random.default_rng(0)
    r = np.zeros(1000)
    for t in range(1, len(r)):
        r[t] = -0.5 * r[t - 1] + 0.005 * rng.standard_normal()
    r[-1] = 0.02  # big up day -> expect a pull-back tomorrow
    y = np.cumsum(r)
    f = AutoRegressive().forecast(y, 3)
    assert f[0] < y[-1]


@pytest.mark.parametrize("model", [RidgeDirect(), PatternMatch(), DampedHolt(), AutoRegressive()], ids=lambda m: m.key)
def test_models_only_use_the_history_they_are_given(model, y):
    # Same prefix -> same forecast, no matter what comes afterwards in the full array.
    cut = 700
    a = model.forecast(y[:cut].copy(), H)
    tampered = y.copy()
    tampered[cut:] += 0.5
    b = model.forecast(tampered[:cut], H)
    assert np.allclose(a, b)


def test_features_do_not_peek_ahead(y):
    full = technical_features(y)
    part = technical_features(y[:500])
    assert np.allclose(full[:500], part, equal_nan=True)


def test_pattern_match_neighbours_end_before_the_query(y):
    m = PatternMatch()
    m.forecast(y, H)
    assert m.last_matches
    assert max(m.last_matches) + H < len(y) - m.window
