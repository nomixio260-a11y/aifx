"""Deep learning direction forecasts, tested walk-forward: do sequence models beat a coin flip?

Research only (needs PyTorch, which the server does not use). Each sample is one pair at one bar's
close (the origin). Inputs: the pair's last 64 hourly bars as a sequence with, per bar, the log return
and the high-low range in units of the pair's trailing volatility (EWMA of squared returns over the
bars before that bar), the same scaled returns of the other 6 pairs at that bar (aligned by time,
the last price at or before the bar, at most 3 hours old), and the bar's New York time (hour and
weekday as sine/cosine, a flag for the 17:00 rollover hour). Static inputs: the pair (an embedding),
the interest-rate difference (base minus quote, as known the day before) and the New York hour,
weekday and rollover flag of the next bar. Target: the sign of the log move from the origin's close to
the close H bars later (H = 1, 4, 24; zero moves are left out), one output per horizon.

Two small networks, pooled over the 7 pairs: a 1-D CNN (two stride-2 layers) feeding a GRU, and a
2-layer Transformer encoder over 4-bar patches with sinusoidal positions. Walk-forward exactly like
research_ml.py: retrained from scratch before each test month on everything before it; a training
sample's target must end before the first test origin (so the gap is at least H bars, counted in
bars, which also holds across weekends); the last 10 % of each training window (in time, with the
same gap) is the early-stopping set, and its predictions set the confidence cut-offs (top 1 %,
2 %, ... by |p - 0.5|) used on the test month. Channel means and deviations come from the training
samples only. The better network on the tune period (mean AUC) goes on to daily bars (60-bar
sequences, weekday only, H = 1, 5, 20, tune 2002-2016, test from 2017, retrained yearly). Both
networks are rerun without any time input (does their skill come only from the time of day?) and,
on the test period, with another random seed.

Scores: AUC, hit rate, Brier skill against always 0.5, mean signed move (bp) with block t statistics
(weekly sums over all pairs for hourly bars, monthly for daily), and the same for the most confident
calls. For comparison, the New York time-of-day drift (as aifx/season.py computes it) at the same
coverage, how many of the networks' confident calls fall on drift calls, and LightGBM from
research/ml.json. Leakage checks: features rebuilt from truncated data must equal the full build,
cross-pair alignment never takes a later bar, the train/test gap in bars, a deliberate leak (target =
the bar that already happened) must show as near-perfect accuracy, and where each bar's close, high
and low lie in time (from finer bars). That last check found a leak: Yahoo's daily bar closes at the start of
its day but its high and low span that day, so daily inputs use the previous bar's range; the report
also shows the test period with the same day's range.

    python -m aifx.research_dl      # writes research/dl.md and research/dl.json (about 45 minutes on 2 CPU threads)
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .data import PAIRS
from .research_ml import _auc

REPORT_DIR = Path("research")
NY = "America/New_York"
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
CODES = list(PAIRS)
CONFIG = {
    # range_lag: bars back to take the high-low range from. Yahoo's daily bar labelled D closes at the price at
    # 00:00 UTC of D but its high and low span the rest of day D (after that close; see timing_check), so the
    # daily range is the previous bar's. An hourly bar's high and low lie within the bar.
    "1h": {"horizons": (1, 4, 24), "seq": 64, "lam": 0.98, "warm": 300, "stale": 3 * HOUR_NS, "range_lag": 0,
           "tune_share": 0.6, "tune_from": "2024-12-01", "retrain": "MS", "block": "W"},
    "1d": {"horizons": (1, 5, 20), "seq": 60, "lam": 0.94, "warm": 100, "stale": 4 * DAY_NS, "range_lag": 1,
           "start": "2002-01-01", "split": "2017-01-01", "tune_from": "2010-01-01", "retrain": "YS", "block": "M"},
}
ARCHS = ("cnn_gru", "transformer")
TOPS = (0.01, 0.02, 0.05, 0.10, 0.20)
TRAIN = {"batch": 512, "epochs": 4, "patience": 1, "lr": 1e-3, "wd": 0.01, "val_share": 0.1, "seed": 7}
THREADS = 2
CLIP = 8.0
DRIFT = {"window": 6000, "min_n": 15, "t_min": 2.0, "t_high": 4.0, "min_bars": 2000}


def _ts(x) -> int:
    t = pd.Timestamp(x)
    return int((t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")).value)


def _date(ns) -> str:
    return str(pd.Timestamp(int(ns), tz="UTC").tz_convert(None))[:16]


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -CLIP, CLIP)


# ------------------------------------------------------------------ data

def _load(tf: str, code: str, until=None) -> pd.DataFrame:
    df = history.load_hourly(code) if tf == "1h" else history.load_daily(code)
    df = df[~df.index.duplicated()].sort_index().dropna(subset=["close"])
    idx = pd.DatetimeIndex(df.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    df = df.set_axis(idx.as_unit("ns"))
    return df if until is None else df[df.index.asi8 < _ts(until)]


def build(tf: str, until=None, range_lag: int | None = None) -> dict:
    """One row per (pair, bar): the bar's input channels, static inputs and future moves (bp).
    Rows of a pair are contiguous and in time order; a sample's sequence is its row and the
    ``seq - 1`` rows before it."""
    cfg = CONFIG[tf]
    hourly = tf == "1h"
    hs, hmax = cfg["horizons"], max(cfg["horizons"])
    base = {}
    for code in CODES:
        df = _load(tf, code, until)
        c = df["close"].to_numpy(float)
        r = np.concatenate([[np.nan], np.diff(np.log(c))])
        # volatility known before the bar: EWMA of squared returns up to the previous bar
        var = pd.Series(r * r).ewm(alpha=1 - cfg["lam"], adjust=False).mean().shift(1).to_numpy()
        base[code] = {"t": df.index.asi8, "c": c, "r": r, "sig": np.sqrt(var),
                      "hl": np.log(df["high"].to_numpy(float) / df["low"].to_numpy(float))}
        lag = cfg["range_lag"] if range_lag is None else range_lag
        if lag:
            base[code]["hl"] = np.concatenate([np.full(lag, np.nan), base[code]["hl"][:-lag]])
    origins = {code: b["t"] + HOUR_NS if hourly else b["t"] for code, b in base.items()}
    # interest rates as known the UTC day before the origin
    days = {code: pd.DatetimeIndex(pd.to_datetime(o, utc=True).tz_convert(None).normalize() - pd.Timedelta(days=1))
            for code, o in origins.items()}
    rates = history.rates_panel(pd.DatetimeIndex(np.unique(np.concatenate([d.to_numpy() for d in days.values()]))))
    rows, align = [], {}
    for p, code in enumerate(CODES):
        pair, b = PAIRS[code], base[code]
        t, c, n, origin = b["t"], b["c"], len(b["t"]), origins[code]
        ch = {"ret": _clip(b["r"] / b["sig"]), "range": np.clip(b["hl"] / b["sig"], 0, CLIP)}
        for other in CODES:
            if other == code:                        # own slot empty: every channel always means the same pair
                ch[f"x_{other}"] = np.zeros(n)
                continue
            o = base[other]
            pos = np.searchsorted(o["t"], t, side="right") - 1          # last bar of the other pair at or before
            src = o["t"][np.clip(pos, 0, None)]
            fresh = (pos >= 0) & (t - src <= cfg["stale"])
            if not (src[pos >= 0] <= t[pos >= 0]).all():
                raise AssertionError("cross-pair alignment took a later bar")
            cq = np.where(fresh, o["c"][np.clip(pos, 0, None)], np.nan)
            sq = np.where(fresh, o["sig"][np.clip(pos, 0, None)], np.nan)
            ch[f"x_{other}"] = _clip(np.concatenate([[np.nan], np.diff(np.log(cq))]) / sq)
            a = align.setdefault(code, {"same_time": 0, "carried": 0, "missing": 0, "max_age_h": 0.0, "later": 0})
            a["same_time"] += int((fresh & (src == t)).sum())
            a["carried"] += int((fresh & (src < t)).sum())
            a["missing"] += int((~fresh).sum())
            a["max_age_h"] = max(a["max_age_h"], float((t - src)[fresh].max() / HOUR_NS) if fresh.any() else 0.0)
            a["later"] += int((src[pos >= 0] > t[pos >= 0]).sum())
        st = {"carry": (rates[pair.base] - rates[pair.quote]).reindex(days[code]).to_numpy()}
        if hourly:
            ny = pd.to_datetime(t, utc=True).tz_convert(NY)
            h, w = ny.hour.to_numpy(), ny.dayofweek.to_numpy()
            ch["hour_sin"], ch["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
            ch["wday_sin"], ch["wday_cos"] = np.sin(2 * np.pi * w / 7), np.cos(2 * np.pi * w / 7)
            ch["roll"] = (h == 17).astype(float)
            nyo = pd.to_datetime(origin, utc=True).tz_convert(NY)             # the next bar's start
            for k in range(24):
                st[f"next_h{k}"] = (nyo.hour.to_numpy() == k).astype(float)
            for k in range(7):
                st[f"next_d{k}"] = (nyo.dayofweek.to_numpy() == k).astype(float)
            st["next_roll"] = (nyo.hour.to_numpy() == 17).astype(float)
        else:
            w = pd.to_datetime(t, utc=True).dayofweek.to_numpy()
            ch["wday_sin"], ch["wday_cos"] = np.sin(2 * np.pi * w / 7), np.cos(2 * np.pi * w / 7)
            for k in range(7):
                st[f"wday{k}"] = (w == k).astype(float)
        fwd = np.full((n, len(hs)), np.nan)
        for k, H in enumerate(hs):
            fwd[:-H, k] = np.log(c[H:] / c[:-H]) * 1e4
        tend = np.full(n, np.iinfo(np.int64).max)       # when the longest target ends (the origin H_max bars on)
        tend[:-hmax] = origin[hmax:]
        naive = pd.to_datetime(origin, utc=True).tz_convert(None)
        rows.append({"ch": ch, "st": st, "fwd": fwd, "back": b["r"] * 1e4, "t": t, "c": c, "origin": origin,
                     "tend": tend, "pos": np.arange(n), "pid": np.full(n, p),
                     "ok": np.arange(n) >= cfg["warm"] + cfg["seq"] - 1,
                     "block": naive.to_period(cfg["block"]).astype(str).to_numpy()})
    chans, snames = list(rows[0]["ch"]), list(rows[0]["st"])
    time_ch = [i for i, k in enumerate(chans) if k.startswith(("hour", "wday", "roll"))]
    time_st = [i for i, k in enumerate(snames) if k != "carry"]
    D = {"tf": tf, "horizons": hs, "seq": cfg["seq"], "chans": chans, "snames": snames, "time_ch": time_ch,
         "time_st": time_st, "cont_st": [snames.index("carry")], "align": align,
         "F": np.concatenate([np.column_stack([r["ch"][k] for k in chans]) for r in rows]).astype(np.float32),
         "S": np.concatenate([np.column_stack([r["st"][k] for k in snames]) for r in rows]).astype(np.float32)}
    for key in ("fwd", "back", "t", "c", "origin", "tend", "pos", "pid", "ok"):
        D[key] = np.concatenate([r[key] for r in rows])
    D["block"] = pd.factorize(np.concatenate([r["block"] for r in rows]))[0]
    return D


# ------------------------------------------------------------------ networks

def _net(arch: str, n_ch: int, n_st: int, n_out: int, seq: int):
    import torch
    import torch.nn.functional as fn
    from torch import nn

    class CnnGru(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(n_ch, 32, 5, padding=2), nn.GELU(),
                                      nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.GELU(),
                                      nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.GELU())
            self.gru = nn.GRU(32, 48, batch_first=True)
            self.emb = nn.Embedding(len(CODES), 4)
            self.head = nn.Sequential(nn.Linear(48 + 4 + n_st, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, n_out))

        def forward(self, x, s, p):
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            _, last = self.gru(h)
            return self.head(torch.cat([last[-1], self.emb(p), s], 1))

    class Block(nn.Module):
        def __init__(self, d: int, heads: int, ff: int):
            super().__init__()
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
            self.heads, self.drop = heads, nn.Dropout(0.1)

        def forward(self, x):
            b, t, d = x.shape
            q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
            a = fn.scaled_dot_product_attention(q, k, v, dropout_p=0.1 if self.training else 0.0)
            x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, t, d)))
            return x + self.drop(self.ff(self.n2(x)))

    class Transformer(nn.Module):
        def __init__(self, d: int = 32, patch: int = 4):
            super().__init__()
            self.patch = patch
            self.inp = nn.Conv1d(n_ch, d, patch, stride=patch)
            tokens = seq // patch
            pos = torch.arange(tokens).unsqueeze(1)
            freq = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
            pe = torch.zeros(tokens, d)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
            self.register_buffer("pe", pe)
            self.enc = nn.Sequential(Block(d, 4, 64), Block(d, 4, 64))
            self.norm = nn.LayerNorm(d)
            self.emb = nn.Embedding(len(CODES), 4)
            self.head = nn.Sequential(nn.Linear(2 * d + 4 + n_st, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, n_out))

        def forward(self, x, s, p):
            x = x[:, x.shape[1] % self.patch:]                  # patches end at the last bar
            h = self.norm(self.enc(self.inp(x.transpose(1, 2)).transpose(1, 2) + self.pe))
            return self.head(torch.cat([h[:, -1], h.mean(1), self.emb(p), s], 1))

    return {"cnn_gru": CnnGru, "transformer": Transformer}[arch]()


def _fit(net, X, S, pid, Y, M, offs, tr: np.ndarray, va: np.ndarray, epochs: int, seed: int) -> list[float]:
    """AdamW on masked BCE; stop when the validation loss does not improve; keep the best epoch."""
    import torch
    import torch.nn.functional as fn
    opt = torch.optim.AdamW(net.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["wd"])
    rng = np.random.default_rng(seed)
    best, state, bad, hist = math.inf, None, 0, []
    for _ in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(tr))
        for s in range(0, len(perm), TRAIN["batch"]):
            ix = perm[s:s + TRAIN["batch"]]
            m = M[ix]
            out = net(X[ix[:, None] + offs], S[ix], pid[ix])
            loss = (fn.binary_cross_entropy_with_logits(out, Y[ix], reduction="none") * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        num = den = 0.0
        with torch.no_grad():
            for s in range(0, len(va), 4096):
                ix = torch.from_numpy(va[s:s + 4096])
                m = M[ix]
                out = net(X[ix[:, None] + offs], S[ix], pid[ix])
                num += float((fn.binary_cross_entropy_with_logits(out, Y[ix], reduction="none") * m).sum())
                den += float(m.sum())
        vl = num / max(den, 1.0)
        hist.append(vl)
        if vl < best:
            best, state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= TRAIN["patience"]:
                break
    net.load_state_dict(state)
    return hist


def _predict(net, X, S, pid, offs, rows: np.ndarray) -> np.ndarray:
    import torch
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), 4096):
            ix = torch.from_numpy(rows[s:s + 4096])
            out.append(torch.sigmoid(net(X[ix[:, None] + offs], S[ix], pid[ix])).numpy())
    return np.concatenate(out) if out else np.zeros((0, 1))


def walk(D: dict, arch: str, edges: list, use_time: bool = True, train_from=None, epochs: int | None = None,
         log=print, tag: str = "", seed: int | None = None) -> dict:
    """Walk-forward: for each test segment [a, b), train from scratch on samples whose longest target
    ends by ``a`` (early stopping on the last 10 % of them), predict the segment. Returns the
    probabilities (NaN outside the segments) and each test sample's confidence cut-offs."""
    import torch
    torch.set_num_threads(THREADS)
    hs = D["horizons"]
    N, nH = len(D["ok"]), len(hs)
    ch = [i for i in range(len(D["chans"])) if use_time or i not in D["time_ch"]]
    sc = [i for i in range(len(D["snames"])) if use_time or i not in D["time_st"]]
    cont = [sc.index(i) for i in D["cont_st"] if i in sc]
    F0, S0, f = D["F"][:, ch], D["S"][:, sc], D["fwd"]
    Y = torch.from_numpy((f > 0).astype(np.float32))
    M = torch.from_numpy((np.isfinite(f) & (f != 0)).astype(np.float32))
    pid = torch.from_numpy(D["pid"].astype(np.int64))
    offs = torch.arange(-D["seq"] + 1, 1)
    origin, tend, ok, pos, pids = D["origin"], D["tend"], D["ok"], D["pos"], D["pid"]
    start = _ts(train_from) if train_from is not None else np.iinfo(np.int64).min
    seed = TRAIN["seed"] if seed is None else seed
    P = np.full((N, nH), np.nan, np.float32)
    THR = np.full((N, nH, len(TOPS)), np.nan, np.float32)
    info = []
    for a, b in edges:
        t0 = time.time()
        test = np.flatnonzero(ok & (origin >= a) & (origin < b))
        if not len(test):
            continue
        cand = np.flatnonzero(ok & (origin >= start) & (tend <= a))
        v = np.sort(origin[cand])[int(len(cand) * (1 - TRAIN["val_share"]))]
        va = cand[origin[cand] >= v]
        tr = cand[tend[cand] <= v]
        # normalisation from the training samples only
        mu, sd = np.nanmean(F0[tr], 0), np.nanstd(F0[tr], 0)
        mu, sd = np.where(np.isfinite(mu), mu, 0.0), np.where(sd > 1e-8, sd, 1.0)
        X = torch.from_numpy(np.nan_to_num(np.clip((F0 - mu) / sd, -10, 10)).astype(np.float32))
        S = S0.astype(np.float64).copy()
        if cont:
            smu, ssd = np.nanmean(S0[tr][:, cont], 0), np.nanstd(S0[tr][:, cont], 0)
            S[:, cont] = (S0[:, cont] - np.nan_to_num(smu)) / np.where(ssd > 1e-8, ssd, 1.0)
        St = torch.from_numpy(np.nan_to_num(np.clip(S, -10, 10)).astype(np.float32))
        torch.manual_seed(seed)
        net = _net(arch, len(ch), len(sc), nH, D["seq"])
        hist = _fit(net, X, St, pid, Y, M, offs, tr, va, epochs or TRAIN["epochs"], seed)
        P[test] = _predict(net, X, St, pid, offs, test)
        pv = _predict(net, X, St, pid, offs, va)
        for h in range(nH):
            THR[test, h, :] = np.quantile(np.abs(pv[:, h] - 0.5), [1 - q for q in TOPS])
        gap = min(int(pos[test[pids[test] == p]].min() - pos[cand[pids[cand] == p]].max())
                  for p in np.unique(pids[test]) if (pids[cand] == p).any())
        vgap = min(int(pos[va[pids[va] == p]].min() - pos[tr[pids[tr] == p]].max())
                   for p in np.unique(pids[va]) if (pids[tr] == p).any())
        if min(gap, vgap) < max(hs):
            raise AssertionError(f"gap {gap}/{vgap} bars < {max(hs)}")
        info.append({"test": [_date(a), _date(b)], "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(test)),
                     "train_end": _date(origin[tr].max()), "val": [_date(origin[va].min()), _date(origin[va].max())],
                     "gap_bars": gap, "val_gap_bars": vgap, "val_loss": [round(x, 5) for x in hist],
                     "best_epoch": int(np.argmin(hist)) + 1, "seconds": round(time.time() - t0, 1)})
        if log:
            log(f"  {tag} {_date(a)[:10]}: train {len(tr):,} val {len(va):,} test {len(test):,} gap {gap} "
                f"epochs {len(hist)} best {np.argmin(hist) + 1} val_loss {min(hist):.5f} ({time.time() - t0:.0f}s)")
    return {"P": P, "THR": THR, "info": info}


# ------------------------------------------------------------------ time-of-day drift (as aifx/season.py)

def _slot_stats(r: np.ndarray, key: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(r)
    r, key = r[ok], key[ok]
    cnt = np.bincount(key, minlength=n).astype(float)
    s1 = np.bincount(key, weights=r, minlength=n)
    s2 = np.bincount(key, weights=r * r, minlength=n)
    use = cnt >= DRIFT["min_n"]
    safe = np.where(use, cnt, 1.0)
    mu = np.where(use, s1 / safe, 0.0)
    sd = np.sqrt(np.maximum(s2 / safe - mu * mu, 0.0))
    return mu, np.where(use & (sd > 0), mu / np.where(sd > 0, sd, 1.0) * np.sqrt(safe), 0.0)


def drift(D: dict) -> np.ndarray:
    """Signed t statistic of the New York weekday-and-hour drift over the next H bars for every hourly
    row (0: no slot is clear; NaN: fewer than ``min_bars`` earlier bars). Slot statistics come from
    the pair's last ``window`` hourly bars that started before the origin's UTC day; a bar uses its
    weekday-and-hour slot if |t| >= 2, else its hour if |t| >= 2, else nothing; the t of a sum of
    bar drifts combines their standard errors (season.combined_t)."""
    hs = D["horizons"]
    out = np.full((len(D["ok"]), len(hs)), np.nan)
    for p in range(len(CODES)):
        seg = np.flatnonzero(D["pid"] == p)
        t, c = D["t"][seg], D["c"][seg]
        n = len(t)
        r = np.concatenate([[np.nan], np.diff(np.log(c)) * 1e4])
        r[1:][np.diff(t) > HOUR_NS] = np.nan                   # a bar after a pause carries its gap
        ny = pd.to_datetime(t, utc=True).tz_convert(NY)
        tod = ny.hour.to_numpy()
        wk = ny.dayofweek.to_numpy() * 24 + tod
        origin = t + HOUR_NS
        cut = np.searchsorted(t, origin - origin % DAY_NS, side="left")    # bars before the origin's UTC day
        ucut, inv = np.unique(cut, return_inverse=True)
        MW, TW = np.zeros((len(ucut), 168)), np.zeros((len(ucut), 168))
        MD, TD = np.zeros((len(ucut), 24)), np.zeros((len(ucut), 24))
        for u, k in enumerate(ucut):
            lo = max(0, k - DRIFT["window"] - 1)
            MW[u], TW[u] = _slot_stats(r[lo + 1:k], wk[lo + 1:k], 168)
            MD[u], TD[u] = _slot_stats(r[lo + 1:k], tod[lo + 1:k], 24)
        valid = origin >= t[min(DRIFT["min_bars"], n - 1)]
        for j, H in enumerate(hs):
            i = np.flatnonzero(valid & (np.arange(n) + H < n))
            J = i[:, None] + np.arange(1, H + 1)[None, :]
            u = inv[i][:, None]
            tw, mw, td, md = TW[u, wk[J]], MW[u, wk[J]], TD[u, tod[J]], MD[u, tod[J]]
            use_w = np.abs(tw) >= DRIFT["t_min"]
            use_d = ~use_w & (np.abs(td) >= DRIFT["t_min"])
            d = np.where(use_w, mw, np.where(use_d, md, 0.0))
            tt = np.where(use_w, tw, np.where(use_d, td, 0.0))
            se = np.where(tt != 0, np.abs(d / np.where(tt != 0, tt, 1.0)), 0.0)
            den = np.sqrt((se * se).sum(1))
            out[seg[i], j] = np.where(den > 0, d.sum(1) / np.where(den > 0, den, 1.0), 0.0)
    return out


# ------------------------------------------------------------------ scoring

def _block_t(x: np.ndarray, block: np.ndarray) -> float | None:
    """t of the mean from block sums (all pairs pooled; neighbouring bars and pairs are correlated)."""
    cnt = np.bincount(block)
    sums = np.bincount(block, weights=x)[cnt > 0]
    if len(sums) < 3:
        return None
    sd = sums.std(ddof=1)
    return float(sums.mean() / sd * math.sqrt(len(sums))) if sd > 0 else None


def _calls(s: np.ndarray, f: np.ndarray, block: np.ndarray, sel: np.ndarray, base: np.ndarray) -> dict:
    """Direction calls ``s`` (+1 / -1) on the rows ``sel`` (within the scored rows ``base``)."""
    m = sel & base
    n = int(m.sum())
    out = {"n": n, "cover": n / max(int(base.sum()), 1)}
    if n < 10:
        return out
    signed, hit = s[m] * f[m], (s[m] * f[m] > 0).astype(float)
    out.update({"hit": float(hit.mean()), "hit_t": _block_t(hit - 0.5, block[m]), "bp": float(signed.mean()),
                "t": _block_t(signed, block[m])})
    return out


def score(D: dict, run: dict, mask: np.ndarray) -> dict:
    """Per horizon: overall scores on the masked rows and the confident subsets."""
    out = {}
    for h, H in enumerate(D["horizons"]):
        p, f = run["P"][:, h].astype(float), D["fwd"][:, h]
        base = mask & np.isfinite(p) & np.isfinite(f) & (f != 0)
        s = np.where(p > 0.5, 1.0, -1.0)
        r = _calls(s, f, D["block"], base, base)
        up = (f[base] > 0).astype(int)
        r.update({"auc": _auc(up, p[base]), "brier_skill": float(1 - np.mean((p[base] - up) ** 2) / 0.25),
                  "base_up": float(up.mean())})
        conf = np.abs(p - 0.5)
        r["top"] = [dict(_calls(s, f, D["block"], conf >= run["THR"][:, h, k], base), top=q) for k, q in enumerate(TOPS)]
        out[str(H)] = r
    return out


def drift_score(D: dict, dt: np.ndarray, mask: np.ndarray, cuts: dict) -> dict:
    out = {}
    for h, H in enumerate(D["horizons"]):
        f = D["fwd"][:, h]
        base = mask & np.isfinite(f) & (f != 0) & np.isfinite(dt[:, h])
        a, s = np.abs(np.nan_to_num(dt[:, h])), np.sign(np.nan_to_num(dt[:, h]))
        r = {"t2": _calls(s, f, D["block"], a >= DRIFT["t_min"], base),
             "t4": _calls(s, f, D["block"], a >= DRIFT["t_high"], base),
             "top": [dict(_calls(s, f, D["block"], (a > 0) & (a >= cuts[str(H)][k]), base), top=q, cut=cuts[str(H)][k])
                     for k, q in enumerate(TOPS)]}
        out[str(H)] = r
    return out


def call_bias(D: dict, run: dict, mask: np.ndarray) -> dict:
    """Per horizon and pair: the share of up calls, the share of up moves and the AUC within the pair
    (a network that always calls one side of a trending pair scores without timing anything)."""
    out = {}
    for h, H in enumerate(D["horizons"]):
        p, f = run["P"][:, h].astype(float), D["fwd"][:, h]
        ok = mask & np.isfinite(p) & np.isfinite(f) & (f != 0)
        out[str(H)] = {}
        for i, code in enumerate(CODES):
            sel = ok & (D["pid"] == i)
            if sel.sum() >= 30:
                out[str(H)][code] = {"n": int(sel.sum()), "call_up": float((p[sel] > 0.5).mean()), "up": float((f[sel] > 0).mean()),
                                     "auc": _auc((f[sel] > 0).astype(int), p[sel])}
    return out


def overlap(D: dict, run: dict, dt: np.ndarray, mask: np.ndarray, h: int = 0) -> list[dict]:
    """The network's confident calls against the drift: the share that fall on a drift call (|t| >= 2),
    the share of those in the same direction, the share whose bar is the hour into the 17:00 New York
    roll or the roll hour, and the network's hit rate on its confident calls away from drift calls."""
    p, f = run["P"][:, h].astype(float), D["fwd"][:, h]
    base = mask & np.isfinite(p) & np.isfinite(f) & (f != 0)
    s = np.where(p > 0.5, 1.0, -1.0)
    dcall = np.abs(np.nan_to_num(dt[:, h])) >= DRIFT["t_min"]
    ny = pd.to_datetime(D["origin"], utc=True).tz_convert(NY).hour.to_numpy()     # the next bar's start
    roll = np.isin(ny, (16, 17))
    out = []
    for k, q in enumerate(TOPS):
        sel = base & (np.abs(p - 0.5) >= run["THR"][:, h, k])
        n, both = int(sel.sum()), sel & dcall
        off = _calls(s, f, D["block"], sel & ~dcall, base)
        out.append({"top": q, "n": n, "on_drift": float(both.sum() / max(n, 1)),
                    "agree": float((both & (np.sign(np.nan_to_num(dt[:, h])) == s)).sum() / max(int(both.sum()), 1)),
                    "near_roll": float((sel & roll).sum() / max(n, 1)),
                    "off_n": off["n"], "off_hit": off.get("hit"), "off_t": off.get("t")})
    return out


def _edges(a: int, b: int, freq: str) -> list[tuple[int, int]]:
    pts = [a] + [_ts(m) for m in pd.date_range(pd.Timestamp(a, tz="UTC").tz_convert(None),
                                               pd.Timestamp(b, tz="UTC").tz_convert(None), freq=freq)
                 if a < _ts(m) < b] + [b]
    return list(zip(pts[:-1], pts[1:]))


def _cached(cache, name: str, fn, *args, **kw):
    """``fn(*args, **kw)``, kept in ``cache`` (a directory; None: no cache) under ``name``."""
    if cache is None:
        return fn(*args, **kw)
    path = Path(cache) / f"{name}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = fn(*args, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


# ------------------------------------------------------------------ leakage checks

def truncation_check(tf: str, D: dict, cut) -> dict:
    """Rebuild the inputs from prices before ``cut`` only: every row they share must be identical
    (no input uses a later bar, volatility and alignment included)."""
    T = build(tf, until=cut)
    worst, rows = 0.0, 0
    for p in range(len(CODES)):
        a, b = np.flatnonzero(D["pid"] == p), np.flatnonzero(T["pid"] == p)
        for key in ("F", "S"):
            x, y = D[key][a[:len(b)]], T[key][b]
            same = np.isnan(x) == np.isnan(y)
            diff = np.abs(np.nan_to_num(x) - np.nan_to_num(y)).max() if len(b) else 0.0
            worst = max(worst, float(diff), 0.0 if same.all() else math.inf)
        rows += len(b)
    return {"cut": str(pd.Timestamp(cut))[:10], "rows_compared": rows, "max_abs_diff": worst}


def _timing(coarse: pd.DataFrame, fine: pd.DataFrame, freq: str, step: pd.Timedelta, min_n: int) -> dict:
    g = fine.resample(freq)
    A = pd.DataFrame({"high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(), "n": g["close"].count()})
    A = A[A["n"] >= min_n]
    idx = coarse.index[coarse.index.isin(A.index) & coarse.index.isin(A.index + step) & coarse.index.isin(A.index - step)]
    c, same, prev, nxt = coarse.loc[idx], A.loc[idx], A.loc[idx - step], A.loc[idx + step]
    out = {"bars": int(len(idx))}
    for key, x, y in (("close_vs_start", c["close"], prev["close"]), ("close_vs_end", c["close"], same["close"]),
                      ("high_vs_before", c["high"], prev["high"]), ("high_vs_own", c["high"], same["high"]),
                      ("high_vs_after", c["high"], nxt["high"]), ("low_vs_own", c["low"], same["low"])):
        out[key] = np.abs(np.log(x.to_numpy() / y.to_numpy())) * 1e4
    return out


def timing_check() -> dict:
    """Where a bar's close, high and low lie in time, from finer bars: daily bars against hourly bars
    (UTC days), hourly bars against 15-minute bars. Median distance (bp, all pairs) of the close from
    the finer price at the start and at the end of the bar's period, and of the high and low from the
    finer high and low over the period before, the bar's own period and the period after."""
    out = {}
    for tf, fine_tf, freq, step, min_n in (("1d", "1h", "D", pd.Timedelta(days=1), 20),
                                           ("1h", "15m", "h", pd.Timedelta(hours=1), 4)):
        parts = []
        for code in CODES:
            fine = history.load_intraday(code, fine_tf)
            fine = fine[~fine.index.duplicated()].sort_index()
            parts.append(_timing(_load(tf, code), fine, freq, step, min_n))
        out[tf] = {"against": fine_tf, "bars": sum(p["bars"] for p in parts)}
        for key in parts[0]:
            if key != "bars":
                out[tf][key] = float(np.median(np.concatenate([p[key] for p in parts])))
    return out


def leak_demo(D: dict, a: int, b: int, log=print) -> dict:
    """A deliberate leak: the target is the bar that has already happened (the last input bar).
    The pipeline must then score near-perfect accuracy; the same tiny run on the real target is the control."""
    L = dict(D)
    L["fwd"] = np.repeat(D["back"][:, None], len(D["horizons"]), 1)
    start = str(pd.Timestamp(a, tz="UTC").tz_convert(None) - pd.Timedelta(days=300))
    mask = (D["origin"] >= a) & (D["origin"] < b)
    out = {}
    for name, data in (("leak", L), ("control", D)):
        run = walk(data, "cnn_gru", [(a, b)], train_from=start, log=log, tag=f"sanity-{name}")
        r = score(data, run, mask)["1"]
        out[name] = {k: r[k] for k in ("n", "hit", "auc", "bp", "t")} | {"n_train": run["info"][0]["n_train"]}
    return out


# ------------------------------------------------------------------ studies

def study_hourly(log=print, cache=None) -> dict:
    tf, cfg = "1h", CONFIG["1h"]
    D = build(tf)
    split = int(np.sort(D["t"])[int(len(D["t"]) * cfg["tune_share"])])     # pooled timeline, like research_direction
    end = int(D["origin"].max()) + 1
    tune_from = _ts(cfg["tune_from"])
    tune_edges, test_edges = _edges(tune_from, split, cfg["retrain"]), _edges(split, end, cfg["retrain"])
    res = {"start": _date(D["origin"][D["ok"]].min()), "split": _date(split), "tune_from": _date(tune_from),
           "end": _date(end), "rows": int(D["ok"].sum()), "channels": D["chans"], "static": D["snames"],
           "retrains": {"tune": len(tune_edges), "test": len(test_edges)}}
    log(f"hourly: {res['rows']:,} samples, split {res['split']}, tune walk-forward from {res['tune_from']}")
    res["checks"] = {"truncation": truncation_check(tf, D, _date(split)), "alignment": D["align"]}
    log(f"truncation check: {res['checks']['truncation']}")
    res["checks"]["leak_demo"] = _cached(cache, "1h_leak", leak_demo, D, *test_edges[0], log=log)
    log(f"leak demo: {res['checks']['leak_demo']}")
    tune_m = D["ok"] & (D["origin"] >= tune_from) & (D["origin"] < split)
    test_m = D["ok"] & (D["origin"] >= split)
    runs, res["models"] = {}, {}
    for arch in ARCHS:
        for period, edges in (("tune", tune_edges), ("test", test_edges)):
            runs[arch, period] = _cached(cache, f"1h_{arch}_{period}", walk, D, arch, edges, log=log,
                                         tag=f"{arch}-{period}")
        res["models"][arch] = {"tune": score(D, runs[arch, "tune"], tune_m), "test": score(D, runs[arch, "test"], test_m),
                               "info": {k: runs[arch, k]["info"] for k in ("tune", "test")}}
        log(f"{arch}: {_brief(res['models'][arch])}")
    better = max(ARCHS, key=lambda a: np.mean([res["models"][a]["tune"][str(H)]["auc"] for H in cfg["horizons"]]))
    res["better"] = better
    # the tune-period choice first, then the other network: the same without any time input, and with another
    # random seed (initial weights, batch order, dropout)
    seed2 = TRAIN["seed"] + 1
    res["seed_check"] = {}
    for arch in sorted(ARCHS, key=lambda a: a != better):
        for period, edges in (("tune", tune_edges), ("test", test_edges)):
            runs[f"{arch}_notime", period] = _cached(cache, f"1h_{arch}_notime_{period}", walk, D, arch, edges,
                                                     use_time=False, log=log, tag=f"{arch}-notime-{period}")
        name = f"{arch}_notime"
        res["models"][name] = {"tune": score(D, runs[name, "tune"], tune_m), "test": score(D, runs[name, "test"], test_m),
                               "info": {k: runs[name, k]["info"] for k in ("tune", "test")}}
        log(f"{arch} without time: {_brief(res['models'][name])}")
        run2 = _cached(cache, f"1h_{arch}_seed{seed2}_test", walk, D, arch, test_edges, log=log,
                       tag=f"{arch}-seed{seed2}-test", seed=seed2)
        res["seed_check"][arch] = {"seed": seed2, "test": score(D, run2, test_m), "info": {"test": run2["info"]}}
    dt = drift(D)
    tune_all = D["ok"] & (D["origin"] < split)
    cuts = {str(H): [float(np.quantile(np.abs(dt[tune_all & np.isfinite(dt[:, h]), h]), 1 - q)) for q in TOPS]
            for h, H in enumerate(cfg["horizons"])}
    res["drift"] = {"cuts": cuts, "tune": drift_score(D, dt, tune_m, cuts), "test": drift_score(D, dt, test_m, cuts),
                    "tune_full": drift_score(D, dt, tune_all, cuts)}
    d1 = res["drift"]["test"]["1"]
    log(f"drift H=1 test: |t|>=2 n={d1['t2']['n']} hit={d1['t2'].get('hit', 0):.3f}; "
        f"|t|>=4 n={d1['t4']['n']} hit={d1['t4'].get('hit', 0):.3f}")
    res["overlap"] = {}
    for (name, period), run in runs.items():
        res["overlap"].setdefault(name, {})[period] = overlap(D, run, dt, tune_m if period == "tune" else test_m)
    return res


def study_daily(arch: str, log=print, cache=None) -> dict:
    tf, cfg = "1d", CONFIG["1d"]
    D = build(tf)
    split, start, tune_from = _ts(cfg["split"]), _ts(cfg["start"]), _ts(cfg["tune_from"])
    end = int(D["origin"].max()) + DAY_NS
    tune_edges, test_edges = _edges(tune_from, split, cfg["retrain"]), _edges(split, end, cfg["retrain"])
    ok = D["ok"] & (D["origin"] >= start)
    res = {"arch": arch, "start": cfg["start"], "split": cfg["split"], "tune_from": cfg["tune_from"], "end": _date(end),
           "rows": int(ok.sum()), "channels": D["chans"], "static": D["snames"],
           "retrains": {"tune": len(tune_edges), "test": len(test_edges)}}
    log(f"daily: {res['rows']:,} samples")
    res["checks"] = {"truncation": truncation_check(tf, D, cfg["split"]), "alignment": D["align"]}
    log(f"truncation check: {res['checks']['truncation']}")
    runs = {period: _cached(cache, f"1d_{arch}_{period}", walk, D, arch, edges, train_from=cfg["start"], log=log,
                            tag=f"daily-{period}")
            for period, edges in (("tune", tune_edges), ("test", test_edges))}
    masks = {"tune": ok & (D["origin"] >= tune_from) & (D["origin"] < split), "test": ok & (D["origin"] >= split)}
    for period, m in masks.items():
        res[period] = score(D, runs[period], m)
    res["bias"] = {period: call_bias(D, runs[period], m) for period, m in masks.items()}
    res["info"] = {k: v["info"] for k, v in runs.items()}
    log(f"daily {arch}: {_brief(res)}")
    # the range as the bar gives it (the day after the close: the leak this study first had), test period only
    D0 = build(tf, range_lag=0)
    run0 = _cached(cache, f"1d_{arch}_samedayrange_test", walk, D0, arch, test_edges, train_from=cfg["start"], log=log,
                   tag="daily-same-day-range-test")
    res["same_day_range"] = {"test": score(D0, run0, D0["ok"] & (D0["origin"] >= split)), "info": {"test": run0["info"]}}
    log("daily with the same day's range: " + "; ".join(f"H={H} auc={x['auc']:.4f} hit={x.get('hit', 0):.4f}"
                                                        for H, x in res["same_day_range"]["test"].items()))
    return res


def _brief(r: dict) -> str:
    return "; ".join(f"{period} H={H} auc={x['auc']:.4f} hit={x.get('hit', 0):.4f} t={x.get('t') or 0:+.2f}"
                     for period in ("tune", "test") for H, x in r[period].items())


def _lgbm() -> dict:
    path = REPORT_DIR / "ml.json"
    if not path.exists():
        return {}
    ml = json.loads(path.read_text(encoding="utf-8"))
    return {tf: {"split": ml[tf]["split"], "h": {H: {k: m["lgbm"].get(k) for k in ("n", "hit", "hit_t", "auc", "pips_t",
                                                                                   "top10_hit")}
                                                  for H, m in ml[tf]["h"].items()}}
            for tf in ("1h", "1d") if tf in ml}


# ------------------------------------------------------------------ report

NAMES = {"cnn_gru": "CNN→GRU", "transformer": "Transformer", "cnn_gru_notime": "CNN→GRU (時間の入力なし)",
         "transformer_notime": "Transformer (時間の入力なし)"}


def _p(x) -> str:
    return "–" if x is None else f"{x:.1%}"


def _t(x) -> str:
    return "–" if x is None else f"{x:+.2f}"


def _train_minutes(res: dict) -> tuple[float, float]:
    """Minutes spent training and predicting (hourly, daily), summed over the retrains."""
    h = sum(i["seconds"] for m in res["1h"]["models"].values() for k in ("tune", "test") for i in m["info"][k])
    h += sum(i["seconds"] for r in res["1h"].get("seed_check", {}).values() for i in r["info"]["test"])
    d = sum(i["seconds"] for k in ("tune", "test") for i in res["1d"]["info"][k])
    d += sum(i["seconds"] for i in res["1d"].get("same_day_range", {}).get("info", {}).get("test", []))
    return h / 60, d / 60


def _n_tests(res: dict) -> int:
    """Test-period results looked at: every network run (seed checks included) x horizon x (overall + subsets)."""
    h = res["1h"]
    runs = len(h["models"]) + len(h.get("seed_check", {}))
    return (runs * len(CONFIG["1h"]["horizons"]) + len(CONFIG["1d"]["horizons"])) * (1 + len(TOPS))


def _overall_rows(name: str, r: dict) -> list[str]:
    L = []
    for period, lab in (("tune", "調整"), ("test", "検証")):
        for H, x in r[period].items():
            L.append(f"| {name} | {H}本後 | {lab} | {x['n']:,} | {_p(x.get('hit'))} ({_t(x.get('hit_t'))}) | {x['auc']:.3f} | "
                     f"{x['brier_skill']:+.4f} | {x.get('bp', 0):+.2f} ({_t(x.get('t'))}) |")
    return L


def _top_rows(name: str, r: dict, period: str) -> list[str]:
    L = []
    for H, x in r[period].items():
        cells = []
        for s in x["top"]:
            cells.append(f"{_p(s.get('hit'))} ({s['cover']:.1%}, {s['n']:,}回, t {_t(s.get('t'))})")
        L.append(f"| {name} | {H}本後 | " + " | ".join(cells) + " |")
    return L


def _best_selective(res: dict) -> list[tuple]:
    """(hit, config, period, H, top, n, t) of every confident subset, best test hit first."""
    out = []
    for name, m in res["1h"]["models"].items():
        for H, x in m["test"].items():
            for s in x["top"]:
                if s.get("hit") is not None:
                    out.append((s["hit"], name, "1h", H, s["top"], s["n"], s.get("t"), s["cover"]))
    for H, x in res["1d"]["test"].items():
        for s in x["top"]:
            if s.get("hit") is not None:
                out.append((s["hit"], res["1d"]["arch"], "1d", H, s["top"], s["n"], s.get("t"), s["cover"]))
    return sorted(out, key=lambda r: -r[0])


def report(res: dict) -> str:
    h, d = res["1h"], res["1d"]
    better = h["better"]
    ntests = _n_tests(res)
    best = _best_selective(res)
    dr = h["drift"]
    lg = res.get("lgbm", {})
    epochs = [i["best_epoch"] for m in h["models"].values() for k in ("tune", "test") for i in m["info"][k]]
    first_epoch = float(np.mean(np.array(epochs) == 1)) if epochs else 0.0
    L = ["# ディープラーニングによる方向の予測 (ウォークフォワード検証)", "",
         "これまでの研究では、29種類の特徴を使った機械学習 (LightGBM など) の方向の的中率は約50%で、"
         "はっきり当たったのは時間帯の偏り (ニューヨーク時間のロールオーバー前後) だけでした ([ml.md](ml.md)、[direction.md](direction.md))。"
         "ここでは、過去の値動きをそのまま時系列として読み込むディープラーニング (1次元CNN+GRU と Transformer) で、"
         "方向を偶然より当てられるか、特に「自信のある予測だけに絞れば70%以上当たるか」を、未来のデータが混ざらない形で確かめました。", "",
         "## 結論", ""]
    L += res["conclusion"] + [""]
    L += ["## 何を試したか", "",
          "- **入力 (1時間足)**: 直近64本の1時間足を時系列として入力しました。各足について、(1) そのペアの値動き (対数リターン) と"
          " (2) 足の値幅 (高値−安値) を、それより前の足だけで計算した値動きの荒さ (指数加重の標準偏差) で割ったもの、"
          "(3) 他の6ペアの同じ時刻の値動き (同じく荒さで割ったもの。時刻でそろえ、欠けた足は最大3時間前までの価格で埋める。後の足は決して使わない)、"
          "(4) ニューヨーク時間の時刻と曜日 (sin/cos) と17時 (ロールオーバー) の印。ほかに、通貨ペアの種類、金利差 (前日までに分かる値)、"
          "次の足のニューヨーク時間の時刻・曜日を入れました。",
          "- **予測するもの**: 予測時点の終値から H 本後 (1時間足は1・4・24本後) の終値までが上か下か。動きがゼロの回は除きます。1つのモデルで3つの予測先を同時に出します。",
          "- **モデル**: (a) 1次元CNN (足を4本分ずつまとめる) → GRU → 全結合、(b) 小さな Transformer (2層、幅32、4本ずつの区切りに位置の情報を加える)。"
          "7ペアをまとめて1つのモデルで学習しました (パラメータ約2〜3万)。損失は2値の交差エントロピー、最適化は AdamW、乱数の種は固定です。",
          "- **学習と検証 (ウォークフォワード)**: 検証期間の毎月の初めに、その時点より前のデータだけで一から学習し直し、その月を予測しました。"
          "学習データの予測先 (最大 H 本後) が予測する月にかからないよう、足の本数で H 本以上の間隔を空けています (週末をまたいでも本数で数えます)。"
          "学習データのうち時間的に最後の10%を「早期終了」用に取り分け (こちらも間隔を空け、混ぜない)、その成績が良くならなくなった時点で学習を止めました "
          f"(学習データを最大4周。選ばれたのは {first_epoch:.0%} の回で1周目で、それ以上学習すると早期終了用データでの成績が悪くなりました)。",
          "- **確信度で絞った的中率**: 予測確率 p が 0.5 から遠いほど自信がある予測です。上位1%・2%・5%・10%・20% の区切りの値は、"
          "テストの月ではなく、その月のモデルの早期終了用データ (過去) の予測から決めました。そのため、実際に絞られる割合は目標から少しずれます (表の「割合」)。",
          f"- **期間**: 1時間足は {h['start'][:10]} 〜 {h['end'][:10]}。全体の時間軸の最初の60%を調整期間、残り40% ({h['split'][:10]} 〜) を検証期間としました "
          f"(research_ml.py と同じ区切り方)。モデルの比較用に、調整期間の後半 ({h['tune_from'][:10]} 〜) も同じ方法で月ごとに予測しました。"
          f"どちらのモデルが良いかは調整期間の成績 (AUC の平均) だけで決め、**{NAMES[better]}** を選びました。",
          "- **成績の見方**: 的中率は方向が当たった割合、AUC は予測確率の順番の正しさ (0.5 = でたらめ)、Brier スキルは「いつも50%と言う予測」と比べた確率の精度 (0 より大きければ良い)、"
          "平均 (bp) は予測した向きの平均の値動き (1bp = 0.01%、コスト抜き)。t 値は、1時間足は週ごと、日足は月ごとに全ペアの結果を合計して計算しました"
          " (隣り合う足やペアの結果は独立ではないため)。t が 2 以上で、偶然では説明しにくい水準です。"
          "的中率の t は、上がる回が多かった期間に「上」と言い続けるだけでも大きくなるため、予測の力は主に AUC と平均の値動きの t で見ます。", ""]
    L += [f"## 1時間足: 全体の成績 (件数 {h['rows']:,})", "",
          "| モデル | 予測先 | 期間 | 件数 | 的中率 (t) | AUC | Brierスキル | 平均 bp (t) |", "|---|---|---|---|---|---|---|---|"]
    for name, m in h["models"].items():
        L += _overall_rows(NAMES[name], m)
    L += ["", "## 1時間足: 自信のある予測に絞った的中率", "",
          "各マスは「的中率 (実際に絞られた割合、回数、平均の値動きの t 値)」です。区切りは過去 (早期終了用データ) で決めています。", ""]
    for period, lab in (("test", "検証期間"), ("tune", "調整期間 (後半)")):
        L += [f"### {lab}", "", "| モデル | 予測先 | " + " | ".join(f"上位{q:.0%}" for q in TOPS) + " |",
              "|---|---|" + "---|" * len(TOPS)]
        for name, m in h["models"].items():
            L += _top_rows(NAMES[name], m, period)
        L.append("")
    archs = [a for a in ARCHS if f"{a}_notime" in h["models"]]
    L += ["## 時間の入力を外すと (アブレーション)", "",
          "時刻・曜日・ロールオーバーの入力をすべて外して、同じ検証をしました (調整期間で選んだモデルと、もう一方のモデルの両方)。"
          "成績が時間の入力ありとほぼ同じなら、モデルの力は時間帯以外から来ています。時間の入力を外して成績が落ちるなら、モデルが使っていたのは時間帯の偏りです。"
          "ただし、値幅や他のペアの動きの並びからも時間帯はある程度わかるため、時間の情報を完全に消せるわけではありません。", "",
          "| モデル | 予測先 | 期間 | 時間あり: 的中率 / AUC | 時間なし: 的中率 / AUC | 時間あり: 上位2%の的中率 | 時間なし: 上位2%の的中率 |",
          "|---|---|---|---|---|---|---|"]
    for arch in archs:
        for period, lab in (("tune", "調整"), ("test", "検証")):
            for H in h["models"][arch][period]:
                a, b = h["models"][arch][period][H], h["models"][f"{arch}_notime"][period][H]
                L.append(f"| {NAMES[arch]} | {H}本後 | {lab} | {_p(a.get('hit'))} / {a['auc']:.3f} | {_p(b.get('hit'))} / {b['auc']:.3f} | "
                         f"{_p(a['top'][1].get('hit'))} ({a['top'][1]['n']:,}回) | {_p(b['top'][1].get('hit'))} ({b['top'][1]['n']:,}回) |")
    sc = h.get("seed_check", {})
    if sc:
        seed2 = next(iter(sc.values()))["seed"]
        L += ["", "## 乱数の種を変えると (検証期間)", "",
              "同じモデルを、乱数の種だけ変えて (初期の重み、学習データの順番、ドロップアウト) もう一度検証しました。"
              "数字が大きく変わるなら、その差は偶然の範囲です。", "",
              f"| モデル | 予測先 | 種 {TRAIN['seed']}: 的中率 / AUC / 上位1% / 上位2% / 上位10% | 種 {seed2}: 的中率 / AUC / 上位1% / 上位2% / 上位10% |",
              "|---|---|---|---|"]
        for arch, r in sc.items():
            for H, a in h["models"][arch]["test"].items():
                L.append(f"| {NAMES[arch]} | {H}本後 | " + " | ".join(
                    f"{_p(x.get('hit'))} / {x['auc']:.3f} / " + " / ".join(f"{_p(x['top'][k].get('hit'))}" for k in (0, 1, 3))
                    for x in (a, r["test"][H])) + " |")
    L += ["", "## 時間帯の偏りとの比較 (同じ割合に絞った場合、検証期間)", "",
          "時間帯の偏り (aifx/season.py と同じ計算: 各ペアの直近6,000本の1時間足で、ニューヨーク時間の曜日×時間ごとの平均の値動きと t 値を、"
          "予測する日の0時 (UTC) より前の足だけから計算。|t| ≥ 2 の枠だけ方向を示す) を、同じ検証期間・同じ足で採点しました。"
          "上位 k% の区切り (偏りの |t| の値) は調整期間で決めています。偏りは全体の約18%の足でしか方向を示さないため、次の1時間の「上位20%」は実際には約18%です。"
          "4本・24本先は、その間の足の偏りを合計した t 値を使いました。各マスは「的中率 (実際の割合、回数、t)」です。"
          "区切りはどちらも過去のデータで決めているため、実際に絞られた割合は偏りとモデルで少し違います。", "",
          "| 予測先 | 絞り方 | 時間帯の偏り | " + " | ".join(NAMES[a] for a in ARCHS) + " |", "|---|---|---|" + "---|" * len(ARCHS)]

    def cell(s):
        return f"{_p(s.get('hit'))} ({s['cover']:.1%}、{s['n']:,}回、{_t(s.get('t'))})"

    for H in dr["test"]:
        x = dr["test"][H]
        for k, q in enumerate(TOPS):
            L.append(f"| {H}本後 | 上位{q:.0%} | {cell(x['top'][k])} | "
                     + " | ".join(cell(h["models"][a]["test"][H]["top"][k]) for a in ARCHS) + " |")
        for key, lab in (("t2", "\\|t\\| ≥ 2"), ("t4", "\\|t\\| ≥ 4")):
            L.append(f"| {H}本後 | 偏り {lab} | {cell(x[key])} |" + " |" * len(ARCHS))
    x1 = dr["tune_full"]["1"]
    L += ["", f"参考: 調整期間全体 (直近2,000本以上の履歴がある足) の時間帯の偏り (次の1時間): |t| ≥ 2 で {_p(x1['t2'].get('hit'))} "
          f"({x1['t2']['n']:,}回)、|t| ≥ 4 で {_p(x1['t4'].get('hit'))} ({x1['t4']['n']:,}回)。"
          "[direction.md](direction.md) の数字とほぼ同じになることで、ここでの再現が正しいことを確かめています。", "",
          "### ディープラーニングの自信のある予測は、時間帯の偏りと同じものか (次の1時間、検証期間)", "",
          "「偏りの足」は時間帯の偏りが方向を示す足 (|t| ≥ 2)、「ロールオーバー前後」は予測する足がニューヨーク時間16時台・17時台の足の割合です。"
          "最後の列は、偏りの足を除いた残りでの的中率です。ここが50%前後なら、モデルが当てているのは時間帯の偏りと同じものです。", "",
          "| モデル | 絞り方 | 回数 | 偏りの足 | うち同じ向き | ロールオーバー前後 | 偏りの足以外での的中率 (回数、t) |", "|---|---|---|---|---|---|---|"]
    for name, ov in h.get("overlap", {}).items():
        for o in ov["test"]:
            L.append(f"| {NAMES[name]} | 上位{o['top']:.0%} | {o['n']:,} | {o['on_drift']:.0%} | {o['agree']:.0%} | {o['near_roll']:.0%} | "
                     f"{_p(o['off_hit'])} ({o['off_n']:,}回、{_t(o['off_t'])}) |")
    L.append("")
    if lg:
        L += ["## LightGBM (29の特徴、[ml.md](ml.md)) との比較 (検証期間)", "",
              "LightGBM の検証期間は少しだけ違います (1時間足は " + lg["1h"]["split"][:10] + " 〜、日足は 2017-01-01 〜)。"
              "「上位10%」は確信度の上位10%の的中率です。", "",
              "| 時間足 | 予測先 | LightGBM: 的中率 / AUC / 上位10% | " + " | ".join(f"{NAMES[a]}: 的中率 / AUC / 上位10%" for a in ARCHS) + " |",
              "|---|---|---|" + "---|" * len(ARCHS)]
        for H in h["models"][better]["test"]:
            m = lg.get("1h", {}).get("h", {}).get(H)
            if m:
                L.append(f"| 1時間足 | {H}本後 | {_p(m['hit'])} / {m['auc']:.3f} / {_p(m['top10_hit'])} | " + " | ".join(
                    f"{_p(g.get('hit'))} / {g['auc']:.3f} / {_p(g['top'][3].get('hit'))}"
                    for g in (h["models"][a]["test"][H] for a in ARCHS)) + " |")
        for H, g in d["test"].items():
            m = lg.get("1d", {}).get("h", {}).get(H)
            if m:
                L.append(f"| 日足 | {H}本後 | {_p(m['hit'])} / {m['auc']:.3f} / {_p(m['top10_hit'])} | " + " | ".join(
                    f"{_p(g.get('hit'))} / {g['auc']:.3f} / {_p(g['top'][3].get('hit'))}" if a == d["arch"] else "–" for a in ARCHS) + " |")
        L.append("")
    L += [f"## 日足 ({NAMES[d['arch']]}、件数 {d['rows']:,})", "",
          f"入力は直近60本の日足 (同じく値動きと1本前の足の値幅を荒さで割ったもの、他の6ペアの値動き、曜日) と、ペアの種類・金利差・曜日。予測先は1・5・20本後。"
          f"学習は {d['start'][:4]} 年から、調整期間の後半 {d['tune_from'][:4]}〜2016 年と検証期間 {d['split'][:4]} 年〜 を、毎年初めに学習し直して予測しました。"
          "日足の終値は 2011年ごろから「その日の始め (0時 UTC) の価格」で、高値・安値はその後のその日1日の値です。"
          "そのため値幅は1本前の足のもの (予測時点までに終わった1日) を使っています (下の「未来のデータが混ざっていないかの確認」を参照)。", "",
          "| 予測先 | 期間 | 件数 | 的中率 (t) | AUC | Brierスキル | 平均 bp (t) |", "|---|---|---|---|---|---|---|"]
    for period, lab in (("tune", "調整"), ("test", "検証")):
        for H, x in d[period].items():
            L.append(f"| {H}本後 | {lab} | {x['n']:,} | {_p(x.get('hit'))} ({_t(x.get('hit_t'))}) | {x['auc']:.3f} | "
                     f"{x['brier_skill']:+.4f} | {x.get('bp', 0):+.2f} ({_t(x.get('t'))}) |")
    L += ["", "| 予測先 | 期間 | " + " | ".join(f"上位{q:.0%}" for q in TOPS) + " |", "|---|---|" + "---|" * len(TOPS)]
    for period, lab in (("test", "検証"), ("tune", "調整")):
        for H, x in d[period].items():
            L.append(f"| {H}本後 | {lab} | " + " | ".join(f"{_p(s.get('hit'))} ({s['cover']:.1%}, {s['n']:,}回, t {_t(s.get('t'))})"
                                                        for s in x["top"]) + " |")
    c1 = np.mean([x["top"][0]["cover"] for x in d["test"].values()])
    if c1 > 2 * TOPS[0]:
        L += ["", f"日足では、区切りを決めた早期終了用データ (直前の1〜2年) より検証の年の方が予測が散らばり、「上位1%」でも実際には平均 {c1:.1%} が選ばれています。"
              "区切りを検証期間で決め直すことはしていません (それをすると未来を使うことになるため)。"]
    if d.get("bias"):
        hs = list(d["test"])
        L += ["", "ペアごとに見ると、モデルが何をしているかがわかります。各マスは「上と予測した割合 / 実際に上がった割合 / そのペアの中での AUC」です"
              " (いつも同じ向きを予測していても、その向きに動き続けた期間なら的中率は上がります。ペアの中での AUC は、その偏りを除いたタイミングの力です)。", "",
              "| 期間 | ペア | " + " | ".join(f"{H}本後" for H in hs) + " |", "|---|---|" + "---|" * len(hs)]
        for period, lab in (("test", "検証"), ("tune", "調整")):
            for code in CODES:
                cells = [d["bias"][period][H].get(code) for H in hs]
                L.append(f"| {lab} | {code} | " + " | ".join(f"{c['call_up']:.0%} / {c['up']:.0%} / {c['auc']:.3f}" if c else "–" for c in cells) + " |")
    ck = h["checks"]
    ld = ck["leak_demo"]
    al = ck["alignment"]
    carried = sum(a["carried"] for a in al.values())
    total = sum(a["same_time"] + a["carried"] + a["missing"] for a in al.values())
    gaps = [i["gap_bars"] for m in h["models"].values() for k in ("tune", "test") for i in m["info"][k]]
    dgaps = [i["gap_bars"] for k in ("tune", "test") for i in d["info"][k]]
    tm = res.get("timing", {})
    sd = d.get("same_day_range", {}).get("test")
    L += ["", "## 未来のデータが混ざっていないかの確認", ""]
    if tm:
        L += [f"- **足の中の時刻 (見つけて直したリーク)**: 日足を1時間足と、1時間足を15分足と比べました。日足の終値は、その日の始め (0時 UTC) の価格とのずれが中央値 "
              f"{tm['1d']['close_vs_start']:.1f} bp (その日の終わりとは {tm['1d']['close_vs_end']:.0f} bp) なのに、高値はその日1日の高値とのずれが "
              f"{tm['1d']['high_vs_own']:.1f} bp (前日とは {tm['1d']['high_vs_before']:.0f} bp) でした。つまり日足の高値・安値は、終値 (予測時点) より後の1日の値です。"
              "最初の版ではこの足の値幅をそのまま入力にしていたため、1本後の値動きと同じ時間の値幅がモデルに見えていました。"
              "そこで日足では1本前の足の値幅を使うように直しました"
              + ("。直す前の検証期間の成績は " + "、".join(f"{H}本後 AUC {x['auc']:.3f} / 的中率 {_p(x.get('hit'))}" for H, x in sd.items())
                 + (" で、直した後 (上の表) とほとんど変わらず、このリークは結果に影響していませんでした"
                    if max(abs(sd[H]["auc"] - d["test"][H]["auc"]) for H in sd) < 0.005 else
                    " で、直した後 (上の表) との差がリークの影響です") if sd else "")
              + f"。1時間足は15分足とのずれが終値 {tm['1h']['close_vs_end']:.1f} bp、高値 {tm['1h']['high_vs_own']:.1f} bp、安値 {tm['1h']['low_vs_own']:.1f} bp で、"
              "高値・安値・終値はすべてその足の中の値でした (この問題はありません)。"
              "この種のずれは下の「途中までのデータで作り直す」確認では見つからないため、別に確かめました。"]
    L += [
          f"- **入力を途中までのデータで作り直す**: 価格データを {ck['truncation']['cut']} より前だけに切って入力を作り直し、全データで作った入力と比べました。"
          f"共通する {ck['truncation']['rows_compared']:,} 行 (1時間足) の最大の差は {ck['truncation']['max_abs_diff']:g}、"
          f"日足 ({d['checks']['truncation']['rows_compared']:,} 行) も {d['checks']['truncation']['max_abs_diff']:g} でした。"
          "つまり、値動きの荒さ (指数加重) の計算、他のペアとの時刻合わせ、時刻の入力のどれも、予測時点より後の足を使っていません。",
          f"- **他のペアとの時刻合わせ**: 常に「その時刻かそれより前の最後の足」を使い、後の足を取った回数は {sum(a['later'] for a in al.values())} 回でした。"
          f"前の足で埋めたのは {carried:,} / {total:,} 回 ({carried / max(total, 1):.2%})、最大 {max(a['max_age_h'] for a in al.values()):.0f} 時間前です。",
          "- **標準化**: 入力の平均と標準偏差は、学習し直すたびにその回の学習データだけから計算しました (早期終了用・テストのデータは使わない)。",
          f"- **学習とテストの間隔**: 学習データの最後とテストの最初の間は、1時間足で最小 {min(gaps)} 本 (必要なのは24本以上)、"
          f"日足で最小 {min(dgaps)} 本 (必要なのは20本以上) でした。早期終了用データとの間も同じ間隔を空けています。",
          f"- **わざと未来を混ぜる確認**: 予測先を「すでに起きた最後の足の向き」(入力に含まれている) にずらして小さく学習すると、"
          f"的中率 {_p(ld['leak'].get('hit'))} (AUC {ld['leak']['auc']:.3f}) になりました。正しい予測先に戻すと {_p(ld['control'].get('hit'))} "
          f"(AUC {ld['control']['auc']:.3f}) です。はっきりした未来の情報が入れば、評価の数字にそのまま表れます。"
          "ただし上の日足の値幅のような弱いリークは、成績が少し良く見える程度なので、この確認だけでは見つかりません。", "",
          "## 注意", "",
          f"- 1時間足で6通り (2つのモデル × 時間の入力あり・なし、乱数の種を変えた2回) × 3つの予測先、日足で3つの予測先、それぞれ全体と5段階の絞り込みで、"
          f"検証期間だけで約{ntests}通りの数字を見ています。"
          "t ≈ 2 の結果が1つ2つ出ても、偶然で説明できます。",
          "- 的中率はコスト (スプレッド) を含みません。ロールオーバー前後はスプレッドが広がり、スワップで値動きが相殺されるため、当たっても利益にはなりにくい時間帯です。",
          "- 学習と予測にかかった時間 (学習し直した回の合計、CPU 2スレッド): 1時間足 {:.0f} 分、日足 {:.0f} 分。".format(*_train_minutes(res)),
          ""]
    L += ["## 確信度で絞った結果の上位 (検証期間)", "",
          "検証期間で的中率が高かった順です。同じモデル・同じ絞り方の、調整期間での的中率と、乱数の種を変えたときの的中率を並べています。", "",
          "| 的中率 | モデル | 時間足 | 予測先 | 絞り方 | 回数 | t | 調整期間 | 種を変えると |", "|---|---|---|---|---|---|---|---|---|"]
    for hit, name, tf, H, q, n, t, cover in best[:10]:
        k = TOPS.index(q)
        tune = (h["models"][name] if tf == "1h" else d)["tune"][H]["top"][k].get("hit")
        other = h.get("seed_check", {}).get(name, {}).get("test", {}).get(H, {}).get("top", [{}] * len(TOPS))[k].get("hit") if tf == "1h" else None
        L.append(f"| {hit:.1%} | {NAMES.get(name, name)} | {'1時間足' if tf == '1h' else '日足'} | {H}本後 | 上位{q:.0%} (実際 {cover:.1%}) | {n:,} | {_t(t)} | "
                 f"{_p(tune)} | {_p(other)} |")
    return "\n".join(L) + "\n"


def conclusion(res: dict) -> list[str]:
    """Plain statements from the numbers (the report's summary)."""
    h, d = res["1h"], res["1d"]
    M, dr, ov = h["models"], h["drift"]["test"], h.get("overlap", {})
    L = []
    # 1. overall
    parts = []
    for a in ARCHS:
        parts.append(f"{NAMES[a]}: " + "、".join(f"{H}本後 {x['auc']:.3f} / {_p(x.get('hit'))} (t {_t(x.get('t'))})"
                                                 for H, x in M[a]["test"].items()))
    tune_sig = [(a, H) for a in ARCHS for H, x in M[a]["tune"].items() if (x.get("t") or 0) >= 2]
    test_sig = [(a, H) for a in ARCHS for H, x in M[a]["test"].items() if (x.get("t") or 0) >= 2]
    L.append(("- **全体の成績は、偶然よりわずかに良い程度です。**" if test_sig else "- **全体の成績は、偶然と区別できません。**")
             + " 検証期間の AUC / 的中率 (平均の値動きの t) は " + "; ".join(parts) + "。"
             + ("AUC 0.5 がでたらめで、LightGBM (1時間足) は {:.3f}〜{:.3f} でした。".format(
                 *(lambda v: (min(v), max(v)))([m["auc"] for m in res["lgbm"]["1h"]["h"].values()])) if res.get("lgbm", {}).get("1h") else "")
             + ("調整期間ではどれも t < 2 で、偶然と区別できません。" if not tune_sig else ""))
    # 2. confident subsets reaching 70 %
    hits70 = [(s["hit"], a, H, k, s) for a in ARCHS for H, x in M[a]["test"].items() for k, s in enumerate(x["top"])
              if (s.get("hit") or 0) >= 0.7 and s["n"] >= 100]
    if hits70:
        hits70.sort(key=lambda r: -r[0])
        items = []
        for hit, a, H, k, s in hits70:
            tu = M[a]["tune"][H]["top"][k]
            o = ov.get(a, {}).get("test", [{}] * len(TOPS))[k] if H == "1" else {}
            items.append(f"{NAMES[a]} の{H}本後・上位{TOPS[k]:.0%} で {hit:.1%} ({s['n']:,}回、t {_t(s.get('t'))}; 調整期間では {_p(tu.get('hit'))})"
                         + (f"。このうち {o['on_drift']:.0%} は時間帯の偏りが方向を示す足で、その {o['agree']:.0%} が偏りと同じ向きでした" if o else ""))
        tune_low = all((M[a]["tune"][H]["top"][k].get("hit") or 0) < 0.7 for _, a, H, k, _ in hits70)
        drift_like = all(H == "1" and ov.get(a, {}).get("test") and ov[a]["test"][k]["on_drift"] >= 0.5
                         and ov[a]["test"][k]["agree"] >= 0.9 for _, a, H, k, _ in hits70)
        L.append("- **自信のある予測に絞ると、検証期間で70%を超えた組み合わせがあります**: " + "; ".join(items) + "。"
                 + ("ただし調整期間では同じ絞り方でも70%に届いていません。" if tune_low else "")
                 + ("中身の大部分は時間帯の偏り (ロールオーバー前後) の再発見です。" if drift_like else ""))
    else:
        best = [b for b in _best_selective(res) if b[5] >= 100]
        b = best[0]
        L.append(f"- **自信のある予測に絞っても、70%に届いた組み合わせはありません** (100回以上のもの)。検証期間で最も高かったのは "
                 f"{NAMES.get(b[1], b[1])} の{'1時間足' if b[2] == '1h' else '日足'} {b[3]}本後・上位{b[4]:.0%} で {b[0]:.1%} ({b[5]:,}回、t {_t(b[6])}) です。")
    # 3. against the drift at the same coverage (next hour)
    rows = []
    beats = []
    for k, q in enumerate(TOPS):
        dd = dr["1"]["top"][k]
        best_a = max(ARCHS, key=lambda a: M[a]["test"]["1"]["top"][k].get("hit") or 0)
        g = M[best_a]["test"]["1"]["top"][k]
        rows.append(f"上位{q:.0%}: 偏り {_p(dd.get('hit'))} / {NAMES[best_a]} {_p(g.get('hit'))}")
        if (g.get("hit") or 0) > (dd.get("hit") or 0):
            beats.append(q)
    L.append(("- **時間帯の偏りには、同じ割合に絞ったどの段階でも及びません。**" if not beats else
              "- 時間帯の偏りとの比較:") + " 次の1時間で、" + "、".join(rows) + " (ディープラーニングは各段階で良い方のモデル)。"
             + ("4本・24本先のディープラーニングの絞った予測は、どれも t が 2 に届きません。" if all(
                 (s.get("t") or 0) < 2 for a in ARCHS for H in ("4", "24") for s in M[a]["test"][H]["top"]) else ""))
    # 4. ablation
    abl = []
    for a in ARCHS:
        n = f"{a}_notime"
        if n in M:
            x, y = M[a]["test"]["1"], M[n]["test"]["1"]
            abl.append(f"{NAMES[a]} は次の1時間の AUC {x['auc']:.3f} → {y['auc']:.3f}、的中率 {_p(x.get('hit'))} → {_p(y.get('hit'))}、"
                       f"上位2% {_p(x['top'][1].get('hit'))} → {_p(y['top'][1].get('hit'))}")
    lost = [a for a in ARCHS if f"{a}_notime" in M
            and M[f"{a}_notime"]["test"]["1"]["auc"] < 0.515 and M[a]["test"]["1"]["auc"] - M[f"{a}_notime"]["test"]["1"]["auc"] > 0.01]
    if abl:
        L.append("- **時間の入力を外すと** (検証期間): " + "; ".join(abl) + "。"
                 + ("次の1時間の力は、時間の入力を外すとほぼ消えます。モデルが使っていたのは時間帯の偏りです"
                    " (わずかに残る AUC 0.51 前後には、値幅などの並びから間接的にわかる時間帯の分も含まれます)。" if len(lost) == len(abl) else ""))
    # 5. seeds
    sc = h.get("seed_check", {})
    if sc:
        fell = [(a, H, k) for _, a, H, k, _ in hits70 if a in sc and (sc[a]["test"][H]["top"][k].get("hit") or 0) < 0.7]
        L.append("- **乱数の種を変えると** (検証期間、次の1時間の上位1% / 2%): " + "; ".join(
            f"{NAMES[a]} は {_p(M[a]['test']['1']['top'][0].get('hit'))} / {_p(M[a]['test']['1']['top'][1].get('hit'))} → "
            f"{_p(r['test']['1']['top'][0].get('hit'))} / {_p(r['test']['1']['top'][1].get('hit'))}" for a, r in sc.items()) + "。"
            + ("**70%を超えた組み合わせは、乱数の種を変えるとどれも70%を下回りました** (同じモデル・同じデータでも、学習の偶然で数字が大きく動きます)。"
               if hits70 and len(fell) == len(hits70) else ""))
    # 6. daily
    dd, dtu = d["test"], d["tune"]
    line = (f"- **日足** ({NAMES[d['arch']]}): 検証期間 ({d['split'][:4]}年〜) は "
            + "、".join(f"{H}本後 AUC {x['auc']:.3f} / {_p(x.get('hit'))} (t {_t(x.get('t'))})" for H, x in dd.items())
            + f"、調整期間 ({d['tune_from'][:4]}〜{int(d['split'][:4]) - 1}年) は AUC " + "、".join(f"{x['auc']:.3f}" for x in dtu.values()) + "。")
    if all(x["auc"] < 0.5 for x in dtu.values()) and all(x["auc"] > 0.5 for x in dd.values()):
        line += "調整期間では 0.5 を下回って (外れる方向で) おり、期間によって向きが変わります。"
    bias = d.get("bias", {}).get("test", {})
    jpy = [bias[H][c]["call_up"] for H in bias for c in bias[H] if c.endswith("JPY")]
    if jpy and np.mean(jpy) >= 0.9:
        line += (f"検証期間の中身の多くは、円のペアでほぼ毎回「上」(円安) と予測すること (平均 {np.mean(jpy):.0%}) で、"
                 f"{d['split'][:4]}年以降は円安が続いたため当たりました。")
    sd = d.get("same_day_range", {}).get("test")
    if sd:
        diff = max(abs(sd[H]["auc"] - dd[H]["auc"]) for H in dd)
        line += ("最初の版では日足の値幅に予測時点より後の1日の値が入っていました (直しました) が、直す前の AUC ("
                 + "、".join(f"{x['auc']:.3f}" for x in sd.values()) + ") と"
                 + ("ほとんど変わらず、このリークは結果に影響していませんでした。" if diff < 0.005 else f"は最大 {diff:.3f} 違いました。"))
    L.append(line)
    L.append(f"- 検証期間で約{_n_tests(res)}通りの数字を見ているため、t ≈ 2 の結果が1つ2つあっても偶然で説明できます。")
    # t >= 2 away from the next hour (hourly 4 and 24 bars, daily), with the tune period's t of the same number
    far = []
    for name, per in [(a, M[a]) for a in ARCHS] + [("1d", d)]:
        for H in per["test"]:
            if name != "1d" and H == "1":
                continue
            for k, (x, y) in enumerate(zip([per["test"][H]] + per["test"][H]["top"], [per["tune"][H]] + per["tune"][H]["top"])):
                if (x.get("t") or 0) >= 2:
                    lab = ("日足 " if name == "1d" else f"{NAMES[name]} の") + f"{H}本後" + (f"・上位{TOPS[k - 1]:.0%}" if k else " 全体")
                    far.append((f"{lab} (t {_t(x.get('t'))}、調整期間 t {_t(y.get('t'))})", (y.get("t") or 0) >= 2))
    if not beats and lost:
        L.insert(0, "- **まとめ: ディープラーニングは、すでに使っている時間帯の偏り (ロールオーバー前後) を、それより弱い形で見つけ直しただけでした。** "
                    "同じ割合に絞って比べると、どの段階でも時間帯の偏りの方がよく当たり、時間の入力を外すと力はほぼ消えます。"
                    + ("4時間先・24時間先・日足では、偶然と区別できる予測力は見つかりませんでした。" if not far else
                       "4時間先・24時間先・日足で平均の値動きの t が 2 を超えたのは " + "、".join(f for f, _ in far)
                       + (" で、調整期間では再現せず、見た数の多さを考えると偶然と区別できません。" if not any(r for _, r in far) else
                          " です。調整期間でも t ≥ 2 のものは、偶然とは言い切れません。")))
    elif not beats:
        L.insert(0, "- **まとめ: 時間帯の偏りを超えるディープラーニングの設定は、見つかりませんでした。**")
    return L


def run(log=print, cache=None) -> dict:
    t0 = time.time()
    res = {"config": {"train": TRAIN, "tops": TOPS, "threads": THREADS, "clip": CLIP, "drift": DRIFT,
                      "hourly": {k: v for k, v in CONFIG["1h"].items() if k != "stale"},
                      "daily": {k: v for k, v in CONFIG["1d"].items() if k != "stale"}}}
    res["timing"] = timing_check()
    log(f"bar timing: {res['timing']}")
    res["1h"] = study_hourly(log=log, cache=cache)
    res["1d"] = study_daily(res["1h"]["better"], log=log, cache=cache)
    res["lgbm"] = _lgbm()
    res["n_test_numbers"] = _n_tests(res)
    res["train_minutes"] = dict(zip(("1h", "1d"), (round(m, 1) for m in _train_minutes(res))))
    res["conclusion"] = conclusion(res)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "dl.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    (REPORT_DIR / "dl.md").write_text(report(res), encoding="utf-8")
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    return res


def _json(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


if __name__ == "__main__":
    run()
