import numpy as np
import pytest

from aifx.engine import MODEL_KEYS, combine, norm_cdf
from aifx.learning import BETA_PRIOR, learn_horizon
from aifx.stats import binom_two_sided, diebold_mariano, scores, wilson

PRIOR = {"mse_z": [1.0] * len(MODEL_KEYS), "k": 1.0, "suu": 0.0, "suv": 0.0}


def samples(n, skill=0.0, noise_scale=1.0, news_effect=0.0, seed=0, model_error=None):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = 10.0
        c0 = rng.normal(0, 3)
        x = rng.uniform(-1, 1)
        a = skill * c0 + news_effect * x * s + rng.normal(0, s * noise_scale)
        m = [0.0] + [c0 + (model_error[i] if model_error else 0) * rng.normal() for i in range(len(MODEL_KEYS) - 1)]
        out.append({"m": m, "a": a, "s": s, "k": 1.0, "g": 0.0, "c0": c0, "c": 0.0, "x": x})
    return out


def test_without_evidence_the_forecaster_assumes_no_skill():
    st = learn_horizon(PRIOR, [], half_life=100)
    assert st.gain == 0.0 and st.beta == BETA_PRIOR and st.k == 1.0
    assert np.allclose(st.weights, 1 / len(MODEL_KEYS))


def test_gain_follows_demonstrated_skill_and_stays_bounded():
    good = learn_horizon(PRIOR, samples(3000, skill=1.0), half_life=1e9)
    none = learn_horizon(PRIOR, samples(3000, skill=0.0), half_life=1e9)
    bad = learn_horizon(PRIOR, samples(3000, skill=-3.0), half_life=1e9)
    assert good.gain > 0.3
    assert abs(none.gain) < 0.1
    assert -0.5 <= bad.gain < 0


def test_interval_scale_learns_to_widen():
    st = learn_horizon(PRIOR, samples(2000, noise_scale=2.0), half_life=1e9)
    assert st.k > 1.5


def test_news_coefficient_learns_from_results():
    st = learn_horizon(PRIOR, samples(4000, news_effect=0.3), half_life=1e9)
    assert st.beta > 0.2


def test_weights_favour_the_more_accurate_model():
    errs = [20.0, 0.0, 0.0, 0.0, 0.0]  # errors for the non-RW models in order: drift is badly wrong
    st = learn_horizon(PRIOR, samples(500, model_error=errs), half_life=1e9)
    assert st.weights[MODEL_KEYS.index("drift")] < min(st.weights[i] for i in range(len(MODEL_KEYS)) if MODEL_KEYS[i] != "drift")


def test_combine_applies_gain_calibration_and_news():
    m = {k: 10.0 for k in MODEL_KEYS}
    out = combine(m, [1 / 6] * 6, sigma_raw=20.0, k=0.5, news=1.0, beta=0.1, gain=0.5)
    assert out["c0"] == pytest.approx(10.0) and out["sigma"] == pytest.approx(10.0)
    assert out["c"] == pytest.approx(0.5 * 10 + 0.1 * 1.0 * 10)
    assert out["p_up"] == pytest.approx(norm_cdf(out["c"] / 10))


def test_wilson_and_binomial():
    lo, hi = wilson(60, 100)
    assert lo < 0.6 < hi and 0.49 < lo < 0.51
    assert binom_two_sided(5, 10) == pytest.approx(1.0)
    assert binom_two_sided(9, 10) < 0.05
    assert binom_two_sided(530, 1000) > 0.05


def test_diebold_mariano_sign():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 500) ** 2
    better = base * 0.7
    stat, p = diebold_mariano(better, base)
    assert stat < 0 and p < 0.01


def test_scores_against_random_walk():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 10, 400)
    c = np.zeros(400)
    s = scores(c, a, np.full(400, 10.0), np.full(400, 0.5))
    assert s["skill"] == pytest.approx(0.0)
    assert s["direction"]["n"] == 0 and s["bss"] == pytest.approx(0.0)
    assert 0.75 < s["coverage"]["80"]["rate"] < 0.85


def test_void_outcomes_are_not_learned_from():
    from aifx.learning import samples_from_ledger
    fc = {"h": 1, "t": "2026-09-22T01:00:00Z", "m": [0.0] * 6, "s": 10.0, "k": 1.0, "g": 0.0, "c0": 0.0, "c": 0.0}
    preds = {1: {"seq": 1, "tf": "1h", "p0": 150.0, "news": {"x": 0.0}, "fc": [fc]},
             2: {"seq": 2, "tf": "1h", "p0": 150.0, "news": {"x": 0.0}, "fc": [dict(fc, t="2026-09-22T02:00:00Z")]}}
    outcomes = [{"seq": 3, "items": [[1, 1, None, None], [2, 1, 150.15, "2026-09-22T02:00:00Z"]]}]
    got = samples_from_ledger(preds, outcomes, "1h")
    assert len(got[1]) == 1 and got[1][0]["a"] > 0
