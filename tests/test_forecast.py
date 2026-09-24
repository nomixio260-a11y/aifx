import json

import numpy as np
import pytest

from aifx.data import get_pair, synthetic_prices
from aifx.forecast import build_bundle, build_report, inverse_mse_weights, run_backtest, volatility_path
from aifx.models import default_models


@pytest.fixture(scope="module")
def report():
    df = synthetic_prices(n=900, seed=7)
    return build_report(get_pair("USDJPY"), df, "synthetic", test_days=120)


def test_report_is_strict_json(report):
    json.dumps(report, allow_nan=False)


def test_bands_are_nested_around_the_centre(report):
    f = report["forecast"]
    c = np.array(f["ensemble"])
    b = {k: (np.array(v["lower"]), np.array(v["upper"])) for k, v in f["bands"].items()}
    assert np.all(b["95"][0] <= b["80"][0]) and np.all(b["80"][0] <= b["50"][0])
    assert np.all(b["50"][0] <= c) and np.all(c <= b["50"][1])
    assert np.all(b["50"][1] <= b["80"][1]) and np.all(b["80"][1] <= b["95"][1])
    # Uncertainty grows with the horizon.
    width = b["80"][1] - b["80"][0]
    assert np.all(np.diff(width) > 0)


def test_weights_sum_to_one(report):
    w = np.array(list(report["weights"].values()))
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-3)  # stored rounded to 4 dp


def test_outlook_probabilities(report):
    for o in report["outlook"]:
        assert 0 <= o["p_up"] <= 1
        assert o["label"] in {"上昇", "下落", "横ばい"}
        assert o["lo80"] <= o["price"] <= o["hi80"]


def test_forecast_dates_are_future_business_days(report):
    dates = np.array(report["forecast"]["dates"], dtype="datetime64[D]")
    assert dates[0] > np.datetime64(report["last_date"])
    assert np.all(np.is_busday(dates))


def test_inverse_mse_falls_back_to_equal_weights_without_history():
    sq = np.full((3, 4, 5), np.nan)
    assert np.allclose(inverse_mse_weights(sq), 1 / 3)


def test_inverse_mse_prefers_the_more_accurate_model():
    sq = np.stack([np.full((10, 2), 1.0), np.full((10, 2), 4.0)])
    w = inverse_mse_weights(sq)
    assert np.allclose(w[:, 0], [0.8, 0.2])


def test_backtest_origins_only_see_their_own_past():
    y = np.log(synthetic_prices(n=700, seed=11)["close"].to_numpy())
    bt = run_backtest(y, default_models(), horizon=10, test_days=100, step=10)
    tampered = y.copy()
    tampered[int(bt.origins[3]) + 1:] += 0.3
    bt2 = run_backtest(tampered, default_models(), horizon=10, test_days=100, step=10)
    for k in bt.keys:
        assert np.allclose(bt.preds[k][:4], bt2.preds[k][:4])
    assert np.allclose(bt.ensemble[:4], bt2.ensemble[:4])


def test_volatility_path_is_increasing():
    y = np.log(synthetic_prices(n=600, seed=2)["close"].to_numpy())
    s = volatility_path(y, 20)
    assert np.all(np.diff(s) > 0)


def test_bundle_lists_models_and_pairs(report):
    b = build_bundle([report])
    assert b["pairs"][0]["pair"] == "USDJPY"
    assert b["models"][-1]["key"] == "ensemble"
