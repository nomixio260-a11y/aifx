"""Forecasting models.

Every model works on log closes and exposes one method,
``forecast(y, horizon) -> np.ndarray`` returning the predicted log close for
steps 1..horizon. Models are refit from scratch on every call, so the same
object can be used for the live forecast and for walk-forward backtests
without any state leaking between origins.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Model:
    key = "base"
    name = "base"
    description = ""
    min_history = 60

    def forecast(self, y: np.ndarray, horizon: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class RandomWalk(Model):
    key = "rw"
    name = "ランダムウォーク"
    description = "最新値がそのまま続くと仮定する基準モデル。為替ではこれに勝つのが難しい。"
    min_history = 2

    def forecast(self, y, horizon):
        return np.full(horizon, y[-1])


class Drift(Model):
    key = "drift"
    name = "ドリフト"
    description = "直近1年の平均変化率がそのまま続くと仮定する。"

    def __init__(self, lookback: int = 250):
        self.lookback = lookback

    def forecast(self, y, horizon):
        window = y[-(self.lookback + 1):]
        mu = (window[-1] - window[0]) / (len(window) - 1)
        return y[-1] + mu * np.arange(1, horizon + 1)


class DampedHolt(Model):
    key = "holt"
    name = "指数平滑 (減衰トレンド)"
    description = "水準とトレンドを指数平滑で推定し、トレンドを徐々に弱めながら延長する (Holt法)。"
    min_history = 60

    ALPHAS = np.linspace(0.05, 1.0, 20)
    BETAS = np.array([0.0, 0.02, 0.05, 0.1, 0.2])
    PHIS = np.array([0.8, 0.9, 0.95, 0.98])

    def __init__(self, fit_window: int = 500):
        self.fit_window = fit_window

    def forecast(self, y, horizon):
        y = y[-self.fit_window:]
        a, b, p = (g.ravel() for g in np.meshgrid(self.ALPHAS, self.BETAS, self.PHIS, indexing="ij"))
        level = np.full(a.shape, y[0])
        trend = np.full(a.shape, np.mean(np.diff(y[: min(10, len(y))])))
        sse = np.zeros(a.shape)
        for obs in y[1:]:
            pred = level + p * trend
            sse += (obs - pred) ** 2
            new_level = a * obs + (1 - a) * pred
            trend = b * (new_level - level) + (1 - b) * p * trend
            level = new_level
        best = int(np.argmin(sse))
        phi = p[best]
        steps = np.cumsum(phi ** np.arange(1, horizon + 1))
        return level[best] + steps * trend[best]


class AutoRegressive(Model):
    key = "ar"
    name = "自己回帰 AR(p)"
    description = "日次リターンを過去のリターンで回帰し、次数pをAICで選ぶ (ARIMA(p,1,0) 相当)。"
    min_history = 120

    def __init__(self, max_p: int = 8, fit_window: int = 750):
        self.max_p = max_p
        self.fit_window = fit_window

    def forecast(self, y, horizon):
        r = np.diff(y[-(self.fit_window + 1):])
        best = None
        n_eff = len(r) - self.max_p  # same sample for every p so AICs are comparable
        target = r[self.max_p:]
        for p in range(1, self.max_p + 1):
            lags = np.column_stack([r[self.max_p - k: len(r) - k] for k in range(1, p + 1)])
            X = np.column_stack([np.ones(n_eff), lags])
            coef, *_ = np.linalg.lstsq(X, target, rcond=None)
            resid = target - X @ coef
            aic = n_eff * np.log(resid @ resid / n_eff) + 2 * (p + 1)
            if best is None or aic < best[0]:
                best = (aic, p, coef)
        _, p, coef = best
        hist = list(r[-p:])
        out = np.empty(horizon)
        level = y[-1]
        for h in range(horizon):
            nxt = coef[0] + sum(coef[k] * hist[-k] for k in range(1, p + 1))
            hist.append(nxt)
            level += nxt
            out[h] = level
        return out


def technical_features(y: np.ndarray) -> np.ndarray:
    """Feature matrix (one row per day) built only from data up to that day.

    Rows before enough history exists contain NaN.
    """
    n = len(y)
    r = np.concatenate([[np.nan], np.diff(y)])
    feats = []

    def lagdiff(k):
        out = np.full(n, np.nan)
        out[k:] = y[k:] - y[:-k]
        return out

    def rolling(a, w, fn):
        out = np.full(n, np.nan)
        if n >= w:
            out[w - 1:] = fn(sliding_window_view(a, w), axis=1)
        return out

    feats.append(r)                                   # 1-day return
    feats.append(lagdiff(5))                          # 1-week momentum
    feats.append(lagdiff(20))                         # 1-month momentum
    feats.append(lagdiff(60))                         # 3-month momentum
    feats.append(y - rolling(y, 20, np.mean))         # distance from 20-day mean
    feats.append(y - rolling(y, 75, np.mean))         # distance from 75-day mean
    vol20 = rolling(np.nan_to_num(r), 20, np.std)
    feats.append(vol20)                               # recent volatility
    feats.append(lagdiff(20) / (vol20 * np.sqrt(20) + 1e-12))  # risk-adjusted momentum
    return np.column_stack(feats)


class RidgeDirect(Model):
    key = "ridge"
    name = "リッジ回帰 (テクニカル特徴量)"
    description = "モメンタム・移動平均乖離・ボラティリティから各日数先のリターンを直接予測する機械学習モデル。"
    min_history = 250

    def __init__(self, fit_window: int = 750, shrink: float = 1.0):
        self.fit_window = fit_window
        self.shrink = shrink

    def forecast(self, y, horizon):
        X_all = technical_features(y)
        n = len(y)
        # Row t is a training example for horizon h only if y[t+h] is known.
        rows = np.arange(max(0, n - 1 - self.fit_window - horizon), n - horizon)
        rows = rows[~np.isnan(X_all[rows]).any(axis=1)]
        Y = np.column_stack([y[rows + h] - y[rows] for h in range(1, horizon + 1)])
        X = X_all[rows]
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
        Xs = (X - mu) / sd
        y_mean = Y.mean(axis=0)
        lam = self.shrink * len(rows)
        B = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Y - y_mean))
        x_now = (X_all[-1] - mu) / sd
        return y[-1] + y_mean + x_now @ B


class PatternMatch(Model):
    key = "knn"
    name = "パターン類似 (k近傍)"
    description = "直近のチャート形状に最も似た過去局面を探し、その後の値動きを平均する。"
    min_history = 300

    def __init__(self, window: int = 20, k: int = 15):
        self.window = window
        self.k = k
        self.last_matches: list[int] = []

    def _shapes(self, y):
        w = self.window
        segs = sliding_window_view(y, w + 1)
        rets = np.diff(segs, axis=1)
        scale = rets.std(axis=1, keepdims=True) + 1e-12
        return rets / scale, scale[:, 0]

    def forecast(self, y, horizon):
        w = self.window
        shapes, scale = self._shapes(y)
        # shapes[i] covers y[i : i+w+1]; its end index is i+w.
        query, q_scale = shapes[-1], scale[-1]
        ends = np.arange(len(shapes)) + w
        usable = ends + horizon < len(y) - w  # future fully known and no overlap with the query
        cand = np.flatnonzero(usable)
        dist = np.sqrt(((shapes[cand] - query) ** 2).mean(axis=1))
        order = cand[np.argsort(dist)]
        picked: list[int] = []
        for i in order:  # skip near-duplicate neighbours from the same episode
            if all(abs(i - j) > w // 2 for j in picked):
                picked.append(int(i))
            if len(picked) == self.k:
                break
        picked_arr = np.array(picked)
        e = ends[picked_arr]
        futures = np.column_stack([y[e + h] - y[e] for h in range(1, horizon + 1)])
        futures *= (q_scale / scale[picked_arr])[:, None]  # rescale to today's volatility
        d = np.sqrt(((shapes[picked_arr] - query) ** 2).mean(axis=1))
        wts = 1.0 / (d + 1e-6)
        self.last_matches = [int(x) for x in e[:5]]
        return y[-1] + (wts[:, None] * futures).sum(axis=0) / wts.sum()


def default_models() -> list[Model]:
    return [RandomWalk(), Drift(), DampedHolt(), AutoRegressive(), RidgeDirect(), PatternMatch()]
