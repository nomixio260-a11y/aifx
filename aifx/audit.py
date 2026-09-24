"""Checks that make it impossible for the forecaster to flatter its own record.

``verify`` (cheap, runs every cycle, covers everything):
  * the hash chain is intact and every data file matches what was committed;
  * each headline's and calendar event's ``fetched_at`` equals the time of the
    batch that committed it, so nothing can be back-dated;
  * every prediction was issued after its origin and before all its targets,
    its targets are exactly the ones the calendar rules produce, and its
    origin price is the committed bar at the origin;
  * every outcome was recorded after its target, uses the latest bar at or
    before the target that had been committed at the time, and matches that
    bar's close exactly;
  * every forecast horizon that has become due has been scored, so bad
    forecasts cannot be skipped.

``audit`` (sampled, heavier): rebuilds a prediction from only the data that
had been committed before it and the learning state implied by earlier
outcomes, reruns the same forecasting code, and compares every number. A
forecaster that had looked at future prices or news would not reproduce.
"""

from __future__ import annotations

import bisect
import json
import math
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PAIRS
from .engine import MODEL_KEYS, TIMEFRAMES, prob_up, target_times
from .forecaster import MAX_ORIGIN_AGE, make_prediction, model_version
from .learning import learn, samples_from_ledger
from .ledger import Ledger, chain_problems, data_file_problems
from .store import PriceStore, parse_price_lines
from .timeutil import iso, london_day_end, parse_iso, utcnow


class _Committed:
    """What each data file contained as of any point in the chain."""

    def __init__(self, root: Path, ledger: Ledger):
        self.root = root
        self.ledger = ledger
        self.problems, self.commits = data_file_problems(root, ledger)
        self._lines: dict[str, list[str]] = {}

    def lines(self, path: str) -> list[str]:
        if path not in self._lines:
            full = self.root / path
            self._lines[path] = full.read_text(encoding="utf-8").splitlines() if full.exists() else []
        return self._lines[path]

    def count_before(self, path: str, seq: int) -> int:
        rows = self.commits.get(path, [])
        seqs = [s for s, _ in rows]
        i = bisect.bisect_left(seqs, seq) - 1
        return rows[i][1] if i >= 0 else 0

    def batch_of_line(self, path: str, line_no: int) -> int | None:
        rows = self.commits.get(path, [])
        i = bisect.bisect_left([n for _, n in rows], line_no)
        return rows[i][0] if i < len(rows) else None

    def line_batches(self, path: str) -> list[int | None]:
        """Batch seq that committed each line (index 0 = line 1), in one pass."""
        rows = self.commits.get(path, [])
        out: list[int | None] = []
        j = 0
        for line_no in range(1, len(self.lines(path)) + 1):
            while j < len(rows) and rows[j][1] < line_no:
                j += 1
            out.append(rows[j][0] if j < len(rows) else None)
        return out

    def prices_before(self, pair: str, tf: str, seq: int) -> pd.DataFrame:
        path = PriceStore.path(pair, tf)
        return parse_price_lines(self.lines(path)[: self.count_before(path, seq)], tf)

    def items_before(self, folder: str, seq: int) -> list[dict]:
        out = []
        for path in sorted(self.commits):
            if path.startswith(folder + "/"):
                for line in self.lines(path)[: self.count_before(path, seq)]:
                    if line.strip():
                        out.append(json.loads(line))
        return out


def _close_enough(a: float, b: float, rel: float = 1e-9) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def verify(root: Path | str) -> dict:
    root = Path(root)
    ledger = Ledger(root).load(check=False)
    recs = ledger.records
    problems: list[dict] = []

    def add(kind: str, msg: str) -> None:
        problems.append({"kind": kind, "msg": msg})

    for msg in chain_problems(recs):
        add("chain", msg)
    com = _Committed(root, ledger)
    for msg in com.problems:
        add("data", msg)
    batches = {r["seq"]: r for r in ledger.of_type("batch")}

    # Headlines and calendar events: fetched_at must equal the committing batch time.
    for path in sorted(com.commits):
        if path.startswith(("news/", "calendar/")):
            owners = com.line_batches(path)
            for i, line in enumerate(com.lines(path), start=1):
                if not line.strip():
                    continue
                b = owners[i - 1]
                item = json.loads(line)
                if b is None or item.get("fetched_at") != batches[b]["at"]:
                    add("time", f"{path}:{i} fetched_at {item.get('fetched_at')} does not match its batch")

    # Hourly bars as committed at the end, with the batch that committed each bar.
    hourly_all: dict[str, tuple[pd.DatetimeIndex, np.ndarray, list[int]]] = {}

    def hourly_upto(pair: str, seq: int):
        if pair not in hourly_all:
            path = PriceStore.path(pair, "1h")
            df = parse_price_lines(com.lines(path)[: com.count_before(path, 10 ** 12)], "1h")
            ends = df.index + pd.Timedelta(hours=1)
            # Line 1 is the header, so bar i sits on line i + 2.
            owners = com.line_batches(path)
            commit_seq = [owners[i + 1] or 10 ** 12 for i in range(len(df))]
            hourly_all[pair] = (ends, df["close"].to_numpy(), commit_seq)
        ends, closes, cseq = hourly_all[pair]
        n = bisect.bisect_left(cseq, seq)  # bars committed strictly before seq (commit seqs are non-decreasing)
        return ends[:n], closes[:n]

    preds = {p["seq"]: p for p in ledger.of_type("prediction")}
    priors = {r["seq"]: r for r in ledger.of_type("prior")}
    outcome_seqs = {r["seq"] for r in ledger.of_type("outcome")}
    seen = set()
    for p in preds.values():
        tag = f"prediction seq {p['seq']} ({p['pair']} {p['tf']} {p['origin']})"
        tf = TIMEFRAMES.get(p["tf"])
        if tf is None or p["pair"] not in PAIRS:
            add("rule", f"{tag}: unknown pair or timeframe")
            continue
        key = (p["pair"], p["tf"], p["origin"])
        if key in seen:
            add("rule", f"{tag}: duplicate prediction for the same origin")
        seen.add(key)
        origin, at = parse_iso(p["origin"]), parse_iso(p["at"])
        if origin > at:
            add("time", f"{tag}: issued before its origin")
        if at - origin > MAX_ORIGIN_AGE[p["tf"]]:
            add("rule", f"{tag}: origin too old when issued")
        expected = [iso(t) for t in target_times(tf, origin)]
        got = [f["t"] for f in p["fc"]]
        if [f["h"] for f in p["fc"]] != list(tf.horizons) or got != expected:
            add("rule", f"{tag}: horizons/targets differ from the calendar rules")
        if any(t <= p["at"] for t in got):
            add("time", f"{tag}: a target was not in the future when issued")
        pr = priors.get(p["prior"])
        if pr is None or pr["tf"] != p["tf"] or p["prior"] >= p["seq"]:
            add("rule", f"{tag}: prior reference invalid")
        if p["learn"] and (p["learn"] not in outcome_seqs or p["learn"] >= p["seq"]):
            add("rule", f"{tag}: learning reference invalid")
        if p["news"]["cut"] != p["origin"]:
            add("rule", f"{tag}: news cutoff is not the origin")
        ends, closes = hourly_upto(p["pair"], p["seq"])
        idx = int(ends.searchsorted(pd.Timestamp(origin), side="right")) - 1
        if idx < 0 or not _close_enough(closes[idx], p["p0"]) or iso(ends[idx].to_pydatetime()) != p["p0_bar"]:
            add("data", f"{tag}: origin price is not the committed bar at the origin")
        for f in p["fc"]:
            if len(f["w"]) != len(MODEL_KEYS) or abs(sum(f["w"]) - 1) > 1e-4:
                add("rule", f"{tag} h={f['h']}: weights do not sum to 1")
            c0 = sum(w * m for w, m in zip(f["w"], f["m"]))
            sig = f["s"] * f["k"]
            c = f["g"] * f["c0"] + f["b"] * p["news"]["x"] * sig
            if abs(c0 - f["c0"]) > 2e-3 or abs(c - f["c"]) > 2e-3 or abs(prob_up(f["c"], sig, f.get("nu")) - f["p"]) > 2e-4:
                add("rule", f"{tag} h={f['h']}: stored numbers are inconsistent")

    scored: dict[tuple[int, int], int] = {}
    n_void = 0
    for o in ledger.of_type("outcome"):
        for pseq, h, actual, bar_end in o["items"]:
            tag = f"outcome seq {o['seq']} (prediction {pseq}, h={h})"
            p = preds.get(pseq)
            f = next((x for x in p["fc"] if x["h"] == h), None) if p else None
            if p is None or pseq >= o["seq"] or f is None:
                add("rule", f"{tag}: refers to an unknown forecast")
                continue
            if (pseq, h) in scored:
                add("rule", f"{tag}: scored twice")
            scored[(pseq, h)] = o["seq"]
            if o["at"] < f["t"]:
                add("time", f"{tag}: scored before its target time")
            ends, closes = hourly_upto(p["pair"], o["seq"])
            t = parse_iso(f["t"])
            origin = parse_iso(p["origin"])
            idx = int(ends.searchsorted(pd.Timestamp(t), side="right")) - 1
            has_bar = idx >= 0 and ends[idx].to_pydatetime() > origin
            if actual is None:
                n_void += 1
                if has_bar:
                    add("rule", f"{tag}: marked void although data existed")
                continue
            if not has_bar or iso(ends[idx].to_pydatetime()) != bar_end or not _close_enough(closes[idx], actual):
                add("data", f"{tag}: actual price is not the committed bar at the target")
            if not len(ends) or ends[-1].to_pydatetime() < t:
                add("time", f"{tag}: scored before the data reached the target")

    last_at = recs[-1]["at"] if recs else ""
    pending = 0
    for p in preds.values():
        ends, _ = hourly_upto(p["pair"], 10 ** 12)
        last_end = iso(ends[-1].to_pydatetime()) if len(ends) else ""
        for f in p["fc"]:
            if (p["seq"], f["h"]) in scored:
                continue
            if f["t"] <= last_at and last_end >= f["t"]:
                add("rule", f"prediction seq {p['seq']} h={f['h']}: due but never scored")
            else:
                pending += 1

    return {
        "ok": not problems,
        "n_problems": len(problems),
        "problems": problems[:100],
        "records": len(recs),
        "head": {"seq": recs[-1]["seq"], "hash": recs[-1]["hash"]} if recs else {"seq": 0, "hash": ""},
        "counts": {"predictions": len(preds), "scored": len(scored) - n_void, "void": n_void, "pending": pending,
                   "batches": len(batches), "priors": len(priors)},
        "checked_at": iso(utcnow()),
    }


def _compare(rec: dict, redo: dict) -> dict:
    diffs = {"p0": abs(rec["p0"] - redo["p0"]), "news_x": abs(rec["news"]["x"] - redo["news"]["x"])}
    worst = {"m": 0.0, "s": 0.0, "w": 0.0, "k": 0.0, "b": 0.0, "g": 0.0, "c": 0.0, "p": 0.0}
    for a, b in zip(rec["fc"], redo["fc"]):
        worst["m"] = max(worst["m"], max(abs(x - y) for x, y in zip(a["m"], b["m"])))
        worst["s"] = max(worst["s"], abs(a["s"] - b["s"]) / max(a["s"], 1e-9))
        worst["w"] = max(worst["w"], max(abs(x - y) for x, y in zip(a["w"], b["w"])))
        worst["k"] = max(worst["k"], abs(a["k"] - b["k"]))
        worst["b"] = max(worst["b"], abs(a["b"] - b["b"]))
        worst["g"] = max(worst["g"], abs(a["g"] - b["g"]))
        worst["c"] = max(worst["c"], abs(a["c"] - b["c"]))
        worst["p"] = max(worst["p"], abs(a["p"] - b["p"]))
        if a.get("nu") != b.get("nu"):
            worst["p"] = max(worst["p"], 1.0)
    diffs.update(worst)
    ok = (diffs["p0"] < 1e-9 and diffs["news_x"] < 1e-6 and worst["m"] < 1e-3 and worst["s"] < 1e-6
          and worst["w"] < 1e-6 and worst["k"] < 1e-6 and worst["b"] < 1e-6 and worst["g"] < 1e-6
          and worst["c"] < 2e-3 and worst["p"] < 2e-4)
    return {"ok": ok, "diffs": {k: float(f"{v:.3g}") for k, v in diffs.items()}}


def reconstruct(ledger: Ledger, com: "_Committed", p: dict, pred_map: dict, outcomes: list[dict]):
    """Rebuild prediction ``p`` from only what had been committed before it."""
    tf = TIMEFRAMES[p["tf"]]
    pair = PAIRS[p["pair"]]
    origin = parse_iso(p["origin"])
    bars = com.prices_before(pair.code, tf.key, p["seq"])
    if tf.key == "1h":
        bars = bars[bars.index + pd.Timedelta(hours=1) <= pd.Timestamp(origin)]
    else:
        bars = bars[np.array([london_day_end(d.date()) <= origin for d in bars.index], dtype=bool)]
    hourly = com.prices_before(pair.code, "1h", p["seq"])
    news_items = com.items_before("news", p["seq"])
    events = com.items_before("calendar", p["seq"])
    prior_rec = ledger.by_seq(p["prior"])
    earlier = {s: q for s, q in pred_map.items() if s < p["seq"]}
    samples = samples_from_ledger(earlier, outcomes, tf.key, upto_seq=p["learn"])
    st = learn(prior_rec, samples, tf.horizons, tf.half_life)
    return make_prediction(tf, pair, bars, hourly, origin, news_items, events, p["prior"], p["learn"], st, p["v"])


def audit(root: Path | str, sample: int = 4, seed: str | None = None) -> dict:
    """Recompute a sample of predictions (the latest ones plus random older ones)."""
    root = Path(root)
    ledger = Ledger(root).load()
    com = _Committed(root, ledger)
    version = model_version()
    preds = ledger.of_type("prediction")
    same = [p for p in preds if p["v"] == version]
    rng = random.Random(seed or (ledger.head[1] if preds else "0"))
    latest = same[-sample:]
    older = same[:-sample]
    chosen = latest + (rng.sample(older, min(sample, len(older))) if older else [])
    pred_map = {p["seq"]: p for p in preds}
    outcomes = ledger.of_type("outcome")
    results = []
    for p in chosen:
        redo, _ = reconstruct(ledger, com, p, pred_map, outcomes)
        res = _compare(p, redo)
        res.update({"seq": p["seq"], "pair": p["pair"], "tf": p["tf"], "origin": p["origin"]})
        results.append(res)
    return {
        "ok": all(r["ok"] for r in results),
        "checked": len(results),
        "skipped_other_version": len(preds) - len(same),
        "version": version,
        "results": results,
        "checked_at": iso(utcnow()),
    }


def external_check(root: Path | str, market, days: int = 20, per_pair: int = 5, seed: int = 0) -> dict:
    """Compare a random sample of stored hourly closes with freshly downloaded data."""
    from .data import PAIRS as ALL
    root = Path(root)
    ledger = Ledger(root).load()
    com = _Committed(root, ledger)
    rng = random.Random(seed)
    rows = []
    cutoff = utcnow()
    for code in ALL:
        stored = com.prices_before(code, "1h", 10 ** 12)
        if not len(stored):
            continue
        recent = stored[stored.index >= pd.Timestamp(cutoff - timedelta(days=days))]
        if not len(recent):
            continue
        try:
            fresh, _ = market.hourly(ALL[code], cutoff, "1mo")
        except Exception as exc:
            rows.append({"pair": code, "error": str(exc)})
            continue
        for ts in rng.sample(list(recent.index), min(per_pair, len(recent))):
            if ts in fresh.index:
                a, b = float(recent.loc[ts, "close"]), float(fresh.loc[ts, "close"])
                rows.append({"pair": code, "time": iso(ts.to_pydatetime()), "stored": a, "fresh": round(b, 6),
                             "ok": abs(a - b) <= 2e-4 * a})
    checked = [r for r in rows if "ok" in r]
    return {"ok": all(r["ok"] for r in checked) if checked else None, "checked": len(checked), "rows": rows,
            "checked_at": iso(utcnow())}


__all__ = ["verify", "audit", "external_check", "math"]
