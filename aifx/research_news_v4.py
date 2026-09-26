"""How well does the headline analyzer read FX headlines, and does a better reading predict anything?

Part A (reading accuracy): 300 real headlines from 2026-06-26 .. 2026-09-26 (GDELT page titles and Google
News with the live server's searches, English and Japanese), labelled by hand (research/news_labels.jsonl:
for each headline the tracked currencies it points up or down, and the rule used), are read by
news_v3.analyze_lexicon (lexicon-v3, the analyzer before) and news.analyze_lexicon (lexicon-v4, adopted). The labels are split by
date: v4 was built by reading only the older half ("dev"); the newer half ("test") is the held-out
check. Scores per currency: direction precision (calls with the right sign / all calls), direction
recall (labelled directions called with the right sign / all labelled directions) and F1; per
headline: exact match (every currency right, including "no call").

Part B (predictive value): the GDELT GKG titles cached by research_news (2026-06-26 .. 2026-09-24, time
= the GDELT batch time) and the Google News titles fetched here (date-only ones used from the end of
their day), read by v3, v4 and FinBERT (when its scores are cached), aggregated per currency exactly as
news.pressures does; pair signal = base minus quote, scored against the next 1/4/24 hourly moves of
the seven pairs, with the origins split by date into an older and a newer half.

Sources (all free; downloads are polite and cached under data/news_eval/, which is not committed):
- Google News RSS with the live server's own search terms (news.SOURCES), week by week with
  ``after:``/``before:``, and the current live feeds (news.SOURCES as they stand).
- The GDELT GKG titles already cached by research_news (data/history/gdelt/gkg/).
- GDELT DOC 2.0 API artlist (titles + seendate), at least ``DOC_MIN_GAP_S`` seconds between requests;
  from the sandbox's shared IP it refused almost every request, so its few titles are only counted.

    python -m aifx.research_news_v4 fetch       # Google News and live feeds (only what is not cached)
    python -m aifx.research_news_v4 fetch-doc   # GDELT DOC API (polite, capped at DOC_MAX_ATTEMPTS)
    python -m aifx.research_news_v4 finbert     # FinBERT scores (needs torch and transformers)
    python -m aifx.research_news_v4             # research/news_v4.md and .json from the cached data
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import news as N
from . import news_v3 as V3
from . import research_news as RN
from .data import CURRENCIES, USER_AGENT

REPORT_DIR = Path("research")
EVAL_DIR = Path("data") / "news_eval"
LABELS = REPORT_DIR / "news_labels.jsonl"
FINBERT_CACHE = EVAL_DIR / "finbert_scores.json"


# ------------------------------------------------------------------ fetching

DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DOC_DIR = EVAL_DIR / "doc"
DOC_MIN_GAP_S = 25.0          # GDELT asks for at most one request every 5 s; the sandbox IP is shared
DOC_MAX_ATTEMPTS = 150        # every DOC request ever sent for this study, refusals included
DOC_ATTEMPTS = EVAL_DIR / "_doc_attempts.jsonl"
DOC_QUERIES = {
    "yen_dollar": "yen dollar",
    "usdjpy": '"USD/JPY"',
    "euro_ecb": "euro dollar ECB",
    "sterling_boe": "sterling pound BoE",
    "aud_rba": '"Australian dollar" RBA',
    "ja_yen_rate": "円相場",
    "ja_dollar_yen": "ドル円",
}
DOC_WINDOWS = (("20260626000000", "20260726000000"), ("20260726000000", "20260826000000"),
               ("20260826000000", "20260926000000"))

GN_DIR = EVAL_DIR / "gnews"
GN_GAP_S = 5.0
GN_START = pd.Timestamp("2026-06-26")
GN_END = pd.Timestamp("2026-09-26")
# the live server's Google News searches, without their "when:1d"
GN_QUERIES = {s[0]: (s[2], s[3]) for s in N.SOURCES if s[0].startswith("gn-")}


def _gn_query(sid: str) -> str:
    q = urllib.parse.parse_qs(urllib.parse.urlparse(GN_QUERIES[sid][0]).query)["q"][0]
    return q.replace(" when:1d", "")


def _log_attempt(path: Path, tag: str, status: int, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": tag, "status": status,
                            "bytes": size}) + "\n")


def _attempts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _get(url: str, timeout: float = 90.0, accept: str = "*/*") -> tuple[int, bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except Exception as exc:
        return 0, str(exc).encode()


def _doc_path(name: str, window: tuple[str, str]) -> Path:
    return DOC_DIR / f"{name}_{window[0][:8]}_{window[1][:8]}.json"


def fetch_doc(log=print) -> dict:
    """GDELT DOC artlist for every query and window not cached yet, one request at a time.

    A refusal ("Please limit requests to one every 5 seconds") is followed by a 30-60 s pause; the whole
    study never sends more than ``DOC_MAX_ATTEMPTS`` requests."""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(n, w) for w in DOC_WINDOWS for n in DOC_QUERIES if not _doc_path(n, w).exists()]
    last = 0.0
    while todo:
        if len(_attempts(DOC_ATTEMPTS)) >= DOC_MAX_ATTEMPTS:
            log(f"doc: attempt budget ({DOC_MAX_ATTEMPTS}) used up; {len(todo)} requests not answered")
            break
        name, win = todo[0]
        params = {"query": DOC_QUERIES[name], "mode": "artlist", "format": "json", "maxrecords": 250,
                  "startdatetime": win[0], "enddatetime": win[1], "sort": "hybridrel"}
        url = DOC_URL + "?" + urllib.parse.urlencode(params)
        time.sleep(max(0.0, DOC_MIN_GAP_S - (time.monotonic() - last)))
        last = time.monotonic()
        status, raw = _get(url, accept="application/json")
        _log_attempt(DOC_ATTEMPTS, name, status, len(raw))
        text = raw.decode("utf-8", "replace")
        if status == 200 and "Please limit requests" not in text[:200]:
            try:
                obj = json.loads(text, strict=False) if text.strip() else {}
            except json.JSONDecodeError:
                obj = {"error": text[:300]}
            _doc_path(name, win).write_text(json.dumps({"url": url, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "response": obj}, ensure_ascii=False), encoding="utf-8")
            log(f"doc {name} {win[0][:8]}: {len(obj.get('articles', []))} articles")
            todo.pop(0)
            random.shuffle(todo)
        else:
            log(f"doc {name} {win[0][:8]}: {status} {text[:50]!r}; waiting")
            time.sleep(30 + random.uniform(0, 30))
    return doc_summary()


def doc_summary() -> dict:
    at = _attempts(DOC_ATTEMPTS)
    ok = sum(a["status"] == 200 for a in at)
    cached = sorted(p.name for p in DOC_DIR.glob("*.json")) if DOC_DIR.exists() else []
    return {"attempts": len(at), "answered": ok, "refused": len(at) - ok, "cached_requests": len(cached),
            "planned_requests": len(DOC_QUERIES) * len(DOC_WINDOWS)}


def _gn_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    edges = list(pd.date_range(GN_START, GN_END, freq="7D")) + [GN_END]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _gn_path(sid: str, a: pd.Timestamp, b: pd.Timestamp) -> Path:
    return GN_DIR / f"{sid}_{a:%Y%m%d}_{b:%Y%m%d}.xml"


def fetch_gnews(log=print) -> dict:
    """Google News RSS for the live searches, one week at a time (at most 100 headlines each), plus the
    current feed of every live source; raw responses are cached."""
    GN_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for a, b in _gn_windows():
        for sid, (_url, lang) in GN_QUERIES.items():
            path = _gn_path(sid, a, b)
            if path.exists():
                continue
            q = f"{_gn_query(sid)} after:{a:%Y-%m-%d} before:{b:%Y-%m-%d}"
            status, raw = _get(N._gnews(q, lang), timeout=30, accept="application/rss+xml, application/xml, text/xml")
            if status == 200:
                path.write_bytes(raw)
                n += 1
            log(f"gnews {sid} {a:%m-%d}: {status} {len(raw)} bytes")
            time.sleep(GN_GAP_S)
    live = EVAL_DIR / "live"
    live.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not any(live.glob("*.xml")):
        for s in N.SOURCES:
            status, raw = _get(s[2], timeout=30, accept="application/rss+xml, application/xml, text/xml")
            if status == 200:
                (live / f"{s[0]}_{stamp}.xml").write_bytes(raw)
            log(f"live {s[0]}: {status} {len(raw)} bytes")
            time.sleep(GN_GAP_S if s[0].startswith("gn-") else 1.0)
    return {"gnews_new": n}


# ------------------------------------------------------------------ headline pool

def _hid(title: str) -> str:
    return N.stable_id(N._norm_key(title))


def pool_doc() -> list[dict]:
    """DOC API titles: [{title, time (seendate), source, lang, domain}] (duplicates removed later)."""
    out = []
    for path in sorted(DOC_DIR.glob("*.json")) if DOC_DIR.exists() else []:
        resp = json.loads(path.read_text(encoding="utf-8")).get("response") or {}
        for a in resp.get("articles", []) or []:
            try:
                t = pd.Timestamp(a["seendate"])
            except (KeyError, ValueError):
                continue
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            lang = "ja" if str(a.get("language", "")).lower().startswith("japan") else "en"
            out.append({"title": N._text(a.get("title", "")), "time": t, "source": "gdelt-doc", "lang": lang,
                        "domain": a.get("domain", ""), "query": path.stem.rsplit("_", 2)[0]})
    return out


def pool_gnews() -> list[dict]:
    """Google News (history by week and the current live feeds) and the other live feeds."""
    out = []
    files = sorted(GN_DIR.glob("*.xml")) if GN_DIR.exists() else []
    live = sorted((EVAL_DIR / "live").glob("*.xml")) if (EVAL_DIR / "live").exists() else []
    src_lang = {s[0]: s[3] for s in N.SOURCES}
    for path in files + live:
        sid = path.stem.split("_")[0]
        try:
            entries = N.parse_feed(path.read_bytes())
        except Exception:
            continue
        for e in entries:
            if e["published"] is None:
                continue
            title = N._clean_title(e["title"], e["publisher"])
            t = pd.Timestamp(e["published"]).tz_convert("UTC")
            out.append({"title": title, "time": t, "source": ("gnews-" if path in files else "live-") + sid,
                        "lang": src_lang.get(sid, "en"), "domain": e["publisher"],
                        "date_only": t.hour == 7 and t.minute == 0 and t.second == 0})
    return out


def pool_gkg() -> list[dict]:
    """GDELT GKG titles cached by research_news (English; time = batch time)."""
    out = []
    for path in sorted(RN.GKG_DIR.glob("2*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for key, b in day.items():
            if b.get("missing"):
                continue
            t = pd.Timestamp(key[:8] + " " + key[8:], tz="UTC")
            for title, domain, _url, _tone, _curs, in_title in b["rows"]:
                out.append({"title": title, "time": t, "source": "gdelt-gkg", "lang": "en",
                            "domain": domain.lower().removeprefix("www."), "fx_title": bool(in_title)})
    return out


def pool() -> list[dict]:
    """Every cached headline once per source group (GKG, DOC, Google News and the live feeds) at the first
    time its normalised title was seen, with its sampling stratum."""
    seen: dict[tuple[str, str], dict] = {}
    for h in sorted(pool_gkg() + pool_doc() + pool_gnews(), key=lambda x: (x["time"], x["source"], x["title"])):
        if not h["title"] or len(h["title"]) < 12:
            continue
        hid = _hid(h["title"])
        src = h["source"]
        group = src if src.startswith("gdelt") else "gnews"
        if (group, hid) in seen:
            continue
        if src == "gdelt-gkg":
            stratum = "gkg_fx" if h.get("fx_title") else "gkg_other"
        elif src == "gdelt-doc":
            stratum = "doc"
        else:
            stratum = "gn_ja" if h["lang"] == "ja" else "gn_en"
        seen[(group, hid)] = {**h, "id": hid, "stratum": stratum}
    return list(seen.values())


# how many headlines of each stratum were drawn for labelling (30 were planned for the DOC API; it had
# answered no request when the sample was drawn, so they went to GKG and Google News English)
SAMPLE_PLAN = {"gkg_fx": 95, "gkg_other": 30, "gn_en": 95, "gn_ja": 80}
SAMPLE_SEED = 20260926
SAMPLE_START = pd.Timestamp("2026-06-26", tz="UTC")


def draw_sample(heads: list[dict] | None = None) -> list[dict]:
    """The headlines to label: a seeded random draw per stratum, no two copies of one story."""
    heads = pool() if heads is None else heads
    by: dict[str, list[dict]] = {}
    for h in heads:
        if h["time"] >= SAMPLE_START:
            by.setdefault(h["stratum"], []).append(h)
    plan = dict(SAMPLE_PLAN)
    out: list[dict] = []
    shingles: list[frozenset] = []
    for stratum in sorted(plan):
        cand = sorted(by.get(stratum, []), key=lambda h: h["id"])
        random.Random(f"{SAMPLE_SEED}-{stratum}").shuffle(cand)
        k = 0
        for h in cand:
            if k >= plan[stratum]:
                break
            sh = N._shingles(h["title"])
            if any(len(sh & s) / max(1, len(sh | s)) >= N.STORY_SIMILARITY for s in shingles):
                continue
            shingles.append(sh)
            out.append(h)
            k += 1
    return sorted(out, key=lambda h: (h["time"], h["id"]))


# ------------------------------------------------------------------ labels and reading accuracy

BASIS = {"m": "move", "p": "policy", "d": "data", "y": "yields", "i": "intervention", "r": "risk"}


def load_labels(path: Path = LABELS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _calls(scores: dict[str, float]) -> dict[str, int]:
    return {c: (1 if v > 0 else -1) for c, v in scores.items() if v and c in CURRENCIES}


def judge(pred: dict[str, int], truth: dict[str, int]) -> str:
    """One headline: ok / should_be_neutral / wrong_sign / extra_currency / missed."""
    if pred == truth:
        return "ok"
    if not truth:
        return "should_be_neutral"
    if any(c in pred and pred[c] != s for c, s in truth.items()):
        return "wrong_sign"
    if any(c not in truth for c in pred):
        return "extra_currency"
    return "missed"


def accuracy(rows: list[dict], preds: list[dict[str, int]], skip_basis: tuple[str, ...] = ()) -> dict:
    """Direction precision / recall / F1 per currency and pooled, and exact match per headline.

    ``skip_basis``: labels resting on these rules (e.g. "risk") are dropped from the truth, and calls
    on those currencies of those headlines are not counted either."""
    per = {c: {"tp": 0, "calls": 0, "truth": 0, "wrong_sign": 0} for c in CURRENCIES}
    exact = 0
    kinds: Counter = Counter()
    for r, p in zip(rows, preds):
        truth = {c: s for c, s in r["labels"].items() if r["basis"].get(c) not in skip_basis}
        dropped = {c for c in r["labels"] if c not in truth}
        p = {c: s for c, s in p.items() if c not in dropped}
        kinds[judge(p, truth)] += 1
        exact += p == truth
        for c in CURRENCIES:
            if c in p:
                per[c]["calls"] += 1
            if c in truth:
                per[c]["truth"] += 1
            if c in p and c in truth:
                if p[c] == truth[c]:
                    per[c]["tp"] += 1
                else:
                    per[c]["wrong_sign"] += 1

    def prf(d):
        prec = d["tp"] / d["calls"] if d["calls"] else None
        rec = d["tp"] / d["truth"] if d["truth"] else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if d["calls"] and d["truth"] else None)
        return {**d, "precision": prec, "recall": rec, "f1": f1}

    tot = {k: sum(per[c][k] for c in CURRENCIES) for k in ("tp", "calls", "truth", "wrong_sign")}
    return {"n": len(rows), "exact": exact, "exact_share": exact / len(rows) if rows else None,
            "judge": dict(kinds), "all": prf(tot), "per": {c: prf(per[c]) for c in CURRENCIES}}


# ------------------------------------------------------------------ FinBERT (optional, local)

FINBERT = "ProsusAI/finbert"


def finbert_scores(titles: list[str], batch: int = 32, threads: int = 2, log=print) -> dict[str, list[float]]:
    """P(positive), P(negative), P(neutral) per English title, by FinBERT on the CPU; cached by title id.

    Needs torch and transformers (not dependencies of the server; research only)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache = json.loads(FINBERT_CACHE.read_text(encoding="utf-8")) if FINBERT_CACHE.exists() else {}
    todo = sorted({t for t in titles if _hid(t) not in cache})
    if not todo:
        return cache
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(FINBERT)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT).eval()
    order = [model.config.id2label[i] for i in range(3)]
    idx = [order.index(k) for k in ("positive", "negative", "neutral")]
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(todo), batch):
            part = todo[i:i + batch]
            enc = tok(part, padding=True, truncation=True, max_length=64, return_tensors="pt")
            p = torch.softmax(model(**enc).logits, -1).numpy()
            for t, row in zip(part, p):
                cache[_hid(t)] = [round(float(row[j]), 4) for j in idx]
            if (i // batch) % 100 == 0:
                log(f"finbert {i + len(part)}/{len(todo)} ({time.time() - t0:.0f}s)")
    FINBERT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    FINBERT_CACHE.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    return cache


def finbert_reading(title: str, lang: str, probs: list[float] | None) -> dict[str, float]:
    """FinBERT's sentiment turned into currency calls: the headline's first-named tracked currency gets
    the sentiment (positive minus negative, when positive or negative is the most likely label) and
    every other named tracked currency the opposite ("yen jumps against the dollar": JPY up, USD down).
    Japanese headlines and headlines without a tracked currency get nothing."""
    if lang != "en" or probs is None:
        return {}
    pos, neg, neu = probs
    if neu >= max(pos, neg):
        return {}
    s = pos - neg
    t = title.lower()
    pairs = list(V3.EN_PAIR.finditer(t))
    at: list[tuple[float, str]] = []
    for p in pairs:
        at += [(p.start(), p.group(1).upper()), (p.start() + 0.5, p.group(2).upper())]
    for st, e, cur in N._entities_en(t, title):
        if cur in CURRENCIES and N.EN_CUR_WORD.fullmatch(t[st:e]) and not any(p.start() <= st < p.end() for p in pairs):
            at.append((st, cur))
    named = list(dict.fromkeys(c for _, c in sorted(at)))
    if not named:
        return {}
    return {c: (s if k == 0 else -s) for k, c in enumerate(named)}


# ------------------------------------------------------------------ predictive value

H = RN.H
B_START = RN.GKG_START + pd.Timedelta(hours=72)       # warm-up for the 48 h lookback
B_END = RN.GKG_END
# Google News history often carries only the date (shown as 07:00 UTC); such a headline is used from
# the end of that day (07:00 UTC the next day), so it cannot leak into the hours before it was written.
GN_DATE_ONLY_DELAY = pd.Timedelta(hours=24)


def readers(fin: dict | None) -> dict:
    """name -> function(title, lang) -> {currency: score}."""
    out = {"v3": lambda t, lang: V3.analyze_lexicon(t, lang)["cur"],
           "v4": lambda t, lang: N.analyze_lexicon(t, lang)["cur"]}
    if fin:
        out["finbert"] = lambda t, lang: {c: round(math.tanh(v / 0.75), 3) for c, v in
                                          finbert_reading(t, lang, fin.get(_hid(t))).items()}
    return out


def build_items(heads: list[dict], read) -> list[dict]:
    """Headlines as the live system would store them (research_news.build_items with another reader):
    usable PUBLISH_DELAY after their time, kept when the reader scores them."""
    out = []
    for h in heads:
        cur = read(h["title"], h["lang"])
        if not cur:
            continue
        pub = h["time"] + (GN_DATE_ONLY_DELAY if h.get("date_only") else pd.Timedelta(0))
        out.append({"id": h["id"], "src": "gdelt", "title": h["title"], "lang": h["lang"], "pub": pub,
                    "published_at": N.iso(pub.to_pydatetime()),
                    "fetched_at": N.iso((pub + RN.PUBLISH_DELAY).to_pydatetime()),
                    "an": {"cur": cur, "top": []}})
    return sorted(out, key=lambda it: (it["published_at"], it["id"]))


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    m = np.isfinite(x) & np.isfinite(y) & (x != 0)
    if m.sum() < 30:
        return {"n": int(m.sum())}
    return {"n": int(m.sum()), "pearson": float(np.corrcoef(x[m], y[m])[0, 1]),
            "spearman": float(pd.Series(x[m]).rank().corr(pd.Series(y[m]).rank()))}


def predictive(P: dict, sigs: dict, split: pd.Timestamp) -> dict:
    """Hit rate, mean signed move and daily-block t (research_news.evaluate) plus the correlation of the
    signal with the forward move, for the older and the newer half."""
    ev = RN.evaluate(P, sigs, split)
    for h in H:
        d = RN.stack(P, sigs, h, split)
        ev[h]["corr"] = {"older": _corr(d["s"][d["tune"]], d["f"][d["tune"]]),
                         "newer": _corr(d["s"][~d["tune"]], d["f"][~d["tune"]])}
    return ev


def study_b(heads_by_set: dict[str, list[dict]], fin: dict | None, log=print) -> dict:
    t0 = time.time()
    P = RN.price_panel(B_START, B_END)
    all_o = np.unique(np.concatenate([p["origin"][np.isfinite(p["fwd"][1])].as_unit("ns").asi8 for p in P.values()]))
    split = pd.Timestamp(all_o[len(all_o) // 2], tz="UTC")
    origins = pd.date_range(B_START, B_END, freq="h")
    out: dict = {"start": str(B_START), "end": str(B_END), "split": str(split), "sets": {}}
    for set_name, heads in heads_by_set.items():
        res = {"headlines": len(heads)}
        for name, read in readers(fin).items():
            items = build_items(heads, read)
            if len(items) < 50:
                continue
            story = RN.story_ids(items)
            press = RN.pressure_panel(items, origins, story)
            sigs = RN.pair_sigs(P, press, origins)
            ev = predictive(P, sigs, split)
            nz = np.concatenate([s for s in sigs.values()])
            res[name] = {"scored": len(items), "eval": ev, "up_share": float(np.mean(nz[nz != 0] > 0)) if (nz != 0).any() else None,
                         "per_cur": {c: int(sum(c in it["an"]["cur"] for it in items)) for c in CURRENCIES},
                         "momentum_corr": RN.momentum_corr(P, sigs)}
            log(f"B {set_name:8s} {name:7s} scored {len(items):6d}  older t {RN._mean_t(ev, 'tune'):+.2f} "
                f"hit {RN._mean_hit(ev, 'tune'):.3f} | newer t {RN._mean_t(ev, 'test'):+.2f} hit {RN._mean_hit(ev, 'test'):.3f} "
                f"({time.time() - t0:.0f}s)")
        out["sets"][set_name] = res
    return out


# ---------------------------------------------------------------------- study

def study_a(fin: dict | None) -> dict:
    """Reading accuracy of v3, v4 (and FinBERT, English only) on the labelled headlines."""
    rows = load_labels()
    read = {"v3": [_calls(V3.analyze_lexicon(r["headline"], r["lang"])["cur"]) for r in rows],
            "v4": [_calls(N.analyze_lexicon(r["headline"], r["lang"])["cur"]) for r in rows]}
    if fin:
        read["finbert"] = [_calls(finbert_reading(r["headline"], r["lang"], fin.get(_hid(r["headline"])))) for r in rows]
    out: dict = {"n": len(rows), "labelled": sum(bool(r["labels"]) for r in rows),
                 "directions": sum(len(r["labels"]) for r in rows),
                 "split_date": min(r["time"] for r in rows if r["split"] == "test"),
                 "counts": {sp: {"headlines": sum(r["split"] == sp for r in rows),
                                 "with_label": sum(r["split"] == sp and bool(r["labels"]) for r in rows),
                                 "directions": sum(len(r["labels"]) for r in rows if r["split"] == sp),
                                 "first": min(r["time"] for r in rows if r["split"] == sp),
                                 "last": max(r["time"] for r in rows if r["split"] == sp),
                                 "strata": dict(Counter(r["stratum"] for r in rows if r["split"] == sp)),
                                 "basis": dict(Counter(b for r in rows if r["split"] == sp for b in r["basis"].values()))}
                            for sp in ("dev", "test")}}
    groups = {"all": lambda r: True, "en": lambda r: r["lang"] == "en", "ja": lambda r: r["lang"] == "ja",
              "gdelt": lambda r: r["source"].startswith("gdelt"), "gnews_en": lambda r: r["stratum"] == "gn_en"}
    acc: dict = {}
    for sp in ("dev", "test"):
        for g, f in groups.items():
            idx = [i for i, r in enumerate(rows) if r["split"] == sp and f(r)]
            sub = [rows[i] for i in idx]
            for name, preds in read.items():
                p = [preds[i] for i in idx]
                acc[f"{sp}|{g}|{name}"] = accuracy(sub, p)
                acc[f"{sp}|{g}|{name}|norisk"] = accuracy(sub, p, ("risk",))
    out["acc"] = acc
    # paired comparison on the held-out half: headlines v4 reads right and v3 wrong, and the reverse
    test = [i for i, r in enumerate(rows) if r["split"] == "test"]
    ok3 = [read["v3"][i] == rows[i]["labels"] for i in test]
    ok4 = [read["v4"][i] == rows[i]["labels"] for i in test]
    b = sum(o4 and not o3 for o3, o4 in zip(ok3, ok4))
    c = sum(o3 and not o4 for o3, o4 in zip(ok3, ok4))
    out["paired_test"] = {"v4_only_right": b, "v3_only_right": c, "both_right": sum(a and b_ for a, b_ in zip(ok3, ok4)),
                          "both_wrong": sum(not a and not b_ for a, b_ in zip(ok3, ok4)),
                          "sign_test_p": _binom_two_sided(b, b + c)}
    out["bootstrap_test"] = bootstrap_f1([rows[i] for i in test], [read["v3"][i] for i in test], [read["v4"][i] for i in test])
    out["rows"] = [{"id": r["id"], "split": r["split"], "lang": r["lang"], "headline": r["headline"], "truth": r["labels"],
                    **{name: read[name][i] for name in read}} for i, r in enumerate(rows)]
    return out


def bootstrap_f1(rows: list[dict], pa: list[dict], pb: list[dict], n: int = 2000, seed: int = 1) -> dict:
    """Pooled direction F1 of b minus a, with a 95 % interval from resampling headlines (paired)."""
    def counts(preds):
        tp = np.array([sum(c in r["labels"] and p[c] == r["labels"][c] for c in p) for r, p in zip(rows, preds)], float)
        return tp, np.array([len(p) for p in preds], float), np.array([len(r["labels"]) for r in rows], float)

    (ta, ca, ua), (tb, cb, ub) = counts(pa), counts(pb)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(rows), size=(n, len(rows)))

    def f1(tp, calls, truth):
        return 2 * tp.sum(-1) / np.maximum(calls.sum(-1) + truth.sum(-1), 1)

    diff = f1(tb[idx], cb[idx], ub[idx]) - f1(ta[idx], ca[idx], ua[idx])
    return {"diff": float(f1(tb, cb, ub) - f1(ta, ca, ua)), "lo": float(np.quantile(diff, 0.025)),
            "hi": float(np.quantile(diff, 0.975)), "share_positive": float(np.mean(diff > 0)), "resamples": n}


def existing_cases(path: Path = Path("tests") / "test_news.py") -> dict:
    """The analyzer examples of tests/test_news.py (written for lexicon-v3), read by v4."""
    import ast

    if not path.exists():
        return {}
    cases, fails = 0, []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not (isinstance(node, ast.FunctionDef) and node.name in ("test_keyword_analysis_directions",
                                                                     "test_headline_reading_fixes")):
            continue
        for c in ast.literal_eval(node.decorator_list[0].args[1]):
            cases += 1
            if node.name == "test_keyword_analysis_directions":
                title, lang, want = c
                got = N.analyze_lexicon(title, lang)["cur"]
                ok = set(got) >= set(want) and all(got[k] * v > 0 for k, v in want.items()) and (bool(want) or got == {})
                ok = ok and not ("円高につながらず" in title and "JPY" in got)
            else:
                title, want = c
                got = N.analyze_lexicon(title, "en")["cur"]
                ok = _calls(got) == want
            if not ok:
                fails.append({"title": title, "v4": got, "want": want})
    return {"cases": cases, "pass": cases - len(fails), "fails": fails}


def _binom_two_sided(k: int, n: int) -> float | None:
    if n == 0:
        return None
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return float(min(1.0, sum(p for p in pk if p <= pk[k] + 1e-12)))


def run(log=print) -> dict:
    t0 = time.time()
    fin = json.loads(FINBERT_CACHE.read_text(encoding="utf-8")) if FINBERT_CACHE.exists() else None
    res: dict = {"analyzers": {"v3": V3.ANALYZER, "v4": N.ANALYZER}, "finbert": bool(fin)}
    res["a"] = study_a(fin)
    log(f"A done ({time.time() - t0:.0f}s)")
    heads = pool()
    gkg = [h for h in heads if h["source"] == "gdelt-gkg"]
    gn = [h for h in heads if h["stratum"].startswith("gn")]
    doc = [h for h in heads if h["source"] == "gdelt-doc" and h["lang"] == "en"]
    res["a"]["existing_cases"] = existing_cases()
    res["a"]["sample_reproduced"] = {h["id"] for h in draw_sample(heads)} == {r["id"] for r in load_labels()}
    res["pool"] = {"gkg": len(gkg), "gnews": len(gn), "gnews_ja": sum(h["lang"] == "ja" for h in gn),
                   "gnews_date_only": sum(bool(h.get("date_only")) for h in gn), "doc_en": len(doc),
                   "doc_all": sum(h["source"] == "gdelt-doc" for h in heads), "doc_api": doc_summary()}
    sets = {"gkg": gkg, "gnews": gn}
    if len(doc) >= 1000:
        sets["doc"] = doc
    res["b"] = study_b(sets, fin, log=log)
    res["runtime_s"] = round(time.time() - t0, 1)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "news_v4.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (REPORT_DIR / "news_v4.md").write_text(report(res), encoding="utf-8")
    return res


# ---------------------------------------------------------------------- report

def _pc(x, d=0):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{d}f}%"


def _f2(x):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.2f}"


def _sg(x, d=2):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.{d}f}"


READER_JA = {"v3": "lexicon-v3 (以前)", "v4": "lexicon-v4 (採用)", "finbert": "FinBERT"}
JUDGE_JA = {"ok": "正しい", "should_be_neutral": "点数をつけるべきでない", "wrong_sign": "向きが逆",
            "extra_currency": "通貨が余分", "missed": "見落とし"}
SET_JA = {"gkg": "GDELT (GKG の見出し)", "gnews": "Google ニュース (本番と同じ検索語)", "doc": "GDELT DOC API"}
HOR_JA = {1: "1時間後", 4: "4時間後", 24: "24時間後"}


def _acc_row(label: str, a: dict) -> str:
    x = a["all"]
    return (f"| {label} | {a['n']} | {a['exact']} ({_pc(a['exact_share'])}) | {x['calls']} | {x['tp']} | {x['wrong_sign']} | "
            f"{_pc(x['precision'])} | {_pc(x['recall'])} ({x['tp']}/{x['truth']}) | {_f2(x['f1'])} |")


ACC_HEAD = ("| 分析 | 見出し | 完全一致 | 方向の判定数 | うち正しい | 向きが逆 | 精度 (precision) | 再現率 (recall) | F1 |\n"
            "|---|---|---|---|---|---|---|---|---|")


def report(res: dict) -> str:
    A, B = res["a"], res["b"]
    acc = A["acc"]
    names = [n for n in ("v3", "v4", "finbert") if f"test|all|{n}" in acc]
    L: list[str] = []
    t3, t4 = acc["test|all|v3"], acc["test|all|v4"]
    d3, d4 = acc["dev|all|v3"], acc["dev|all|v4"]
    L += ["# ニュース見出しの読み取りの改善 (lexicon-v4) と、ニュースの予測力の再確認", ""]
    L += ["## 結論", ""]
    L += [f"- **見出しの読み取りは、手で正解をつけた見出しで確かめて改善しました。** 修正を作るのに使わなかった新しい半分 "
          f"({A['counts']['test']['first'][:10]}〜{A['counts']['test']['last'][:10]} の {t3['n']} 本) で、方向の F1 は "
          f"以前の lexicon-v3 の {_f2(t3['all']['f1'])} から lexicon-v4 の {_f2(t4['all']['f1'])} に上がりました "
          f"(精度 {_pc(t3['all']['precision'])} → {_pc(t4['all']['precision'])}、再現率 {_pc(t3['all']['recall'])} → "
          f"{_pc(t4['all']['recall'])}、全通貨が正しい見出し {t3['exact']} → {t4['exact']} 本)。"
          f"v4 だけが正しい見出しは {A['paired_test']['v4_only_right']} 本、v3 だけが正しい見出しは "
          f"{A['paired_test']['v3_only_right']} 本です (符号検定 p = {A['paired_test']['sign_test_p']:.3g})。"
          f"F1 の差 {A['bootstrap_test']['diff']:+.2f} の 95% 区間 (見出しの再抽出 {A['bootstrap_test']['resamples']:,} 回) は "
          f"{A['bootstrap_test']['lo']:+.2f}〜{A['bootstrap_test']['hi']:+.2f} です。"]
    L += [f"- 修正を作った古い半分 (dev、{d3['n']} 本) では F1 {_f2(d3['all']['f1'])} → {_f2(d4['all']['f1'])} です。"
          "この差は作った本人のデータなので有利に出ます。公平な比較は上の新しい半分の数字です。"]
    L += _predict_summary(res)
    if "finbert" in names:
        tf, t3e, t4e = (acc[f"test|en|{n}"]["all"] for n in ("finbert", "v3", "v4"))
        worse = (tf["f1"] or 0) < (t4e["f1"] or 0)
        L += [f"- **FinBERT (ProsusAI/finbert、手元の CPU で実行) も試しました。** 英語の新しい半分で方向の F1 {_f2(tf['f1'])} "
              f"(精度 {_pc(tf['precision'])}、再現率 {_pc(tf['recall'])})、v3 {_f2(t3e['f1'])}、v4 {_f2(t4e['f1'])} です。"
              + ("通貨の向きの読み取りは v4 に及びません。" if worse else "v4 と同程度以上でした。")
              + "FinBERT は文の肯定・否定を判定するモデルで、「どの通貨が上がるか」は分かりません (「金が上昇、ドル安で」は"
              "肯定的でもドルには下向き)。予測力 (結果B) も v3・v4 と同じくありませんでした。"]
    L += ["- **採用:** aifx/news.py の見出し分析を lexicon-v4 に置き換えました (読み取りが良くなり、予測力は"
          "どちらも「なし」で悪化もないため)。予測への重み (学習の初期値 BETA_PRIOR = 0) は変えません。詳しくは最後の節。", ""]

    # ---- data
    P = res["pool"]
    L += ["## データ", ""]
    L += [f"- **GDELT GKG の見出し**: research_news が保存済みの GDELT の公開ファイル (1時間に1回分) から、5通貨に関係する"
          f"記事の見出し {P['gkg']:,} 本 (重複を除く、2026-06-26〜09-24、英語)。時刻は GDELT のまとまりの時刻。",
          f"- **Google ニュース**: 本番 (aifx/news.py の SOURCES) と同じ検索語7つ (日本語4・英語3) を1週間ごとに"
          f" (after:/before:)、各回最大100本、と現在の本番のフィード全部 (ForexLive、各中央銀行を含む): {P['gnews']:,} 本 "
          f"(うち日本語 {P['gnews_ja']:,} 本)。古い記事は日付だけのものが多く ({P['gnews_date_only']:,} 本、時刻が 07:00 UTC と"
          "表示される)、予測力の検証ではその日の終わり (翌日 07:00 UTC) から使えることにしました。",
          f"- **GDELT DOC API**: 指示どおり、7つの検索語 × 3か月 (1か月ずつ) の 21 回を、25秒以上の間隔と、断られたら30〜60秒"
          f"待つ形で試しました。{P['doc_api']['attempts']} 回送って応答は {P['doc_api']['answered']} 回 (残りは「5秒に1回まで」の"
          f"制限で拒否。前回の研究と同じく、この環境の出口 IP は共有されています)。得られた見出しは {P['doc_all']} 本 (英語 "
          f"{P['doc_en']} 本) と少なく、ラベル付けと予測力の検証には使っていません。日本語の検索 (「円相場」) の応答は、"
          "為替と関係のない英語などの記事だけでした (GDELT には日本語のニュースがほとんどありません。research/news.md と同じ)。",
          "- **価格**: 7ペアの1時間足 (Yahoo Finance、history.load_hourly、2026-09-24 08:00 UTC まで)。",
          "- 取得はすべて無料・公開のもので、分析はすべて手元で実行しています (有料 API・クラウドの LLM は不使用)。"
          "生の応答は data/news_eval/ (git の対象外) に保存し、`python -m aifx.research_news_v4` は保存済みのデータだけで"
          "このレポートを作り直します。", ""]
    c = A["counts"]
    L += ["### 正解ラベルをつけた見出し", ""]
    L += [f"{A['n']} 本を無作為に選びました (層ごとに固定の乱数、同じ記事の転載は1本): GDELT の見出しのうち通貨・中銀の名前が"
          "あるもの 95 本とないもの 30 本、Google ニュースの英語 95 本と日本語 80 本。英語と日本語、為替の専門記事と一般記事の"
          "両方で、「点数をつけるべきでない見出し」も含めて読み取りを測れるようにしています。",
          "",
          "| 半分 | 期間 | 見出し | 向きのラベルがある見出し | 通貨の向き (延べ) | 層 | ラベルの根拠 (延べ) |",
          "|---|---|---|---|---|---|---|"]
    for sp, lab in (("dev", "古い半分 (修正を作った)"), ("test", "新しい半分 (検証、修正後に1回だけ採点)")):
        x = c[sp]
        L += [f"| {lab} | {x['first'][:10]}〜{x['last'][:10]} | {x['headlines']} | {x['with_label']} | {x['directions']} | "
              f"{', '.join(f'{k} {v}' for k, v in sorted(x['strata'].items()))} | "
              f"{', '.join(f'{k} {v}' for k, v in sorted(x['basis'].items()))} |"]
    L += ["", "ラベルは research/news_labels.jsonl にあります (見出し・時刻・取得元・通貨ごとの向きと根拠)。"
          "ラベルは v3・v4 の出力を見る前に、見出しだけを読んで付けました。", ""]
    L += LABEL_RULES + [""]

    # ---- accuracy
    L += ["## 結果A: 見出しの読み取りの正確さ", ""]
    L += ["通貨ごと・見出しごとに、正解の向き (+1/−1/なし) と分析の点数の符号を比べました。**精度** = 分析が向きをつけた"
          "うち正しかった割合、**再現率** = 正解の向きのうち分析が正しく当てた割合、**完全一致** = 5通貨すべて (「なし」を"
          "含む) が正解と同じ見出しの割合です。", ""]
    for sp, lab in (("test", "新しい半分 (検証)"), ("dev", "古い半分 (修正を作った期間。v4 に有利)")):
        L += [f"### {lab}", "", ACC_HEAD]
        for n in names:
            if n == "finbert":
                continue
            L += [_acc_row(READER_JA[n], acc[f"{sp}|all|{n}"])]
        L += [""]
    L += ["### 新しい半分の内訳", "", "| 区分 | 分析 | 見出し | 完全一致 | 判定数 | 精度 | 再現率 | F1 |", "|---|---|---|---|---|---|---|---|"]
    for g, lab in (("en", "英語"), ("ja", "日本語"), ("gdelt", "GDELT の見出し"), ("gnews_en", "Google ニュース英語")):
        for n in names:
            if n == "finbert" and g not in ("en", "gdelt", "gnews_en"):
                continue
            a = acc[f"test|{g}|{n}"]
            L += [f"| {lab} | {READER_JA[n]} | {a['n']} | {a['exact']} | {a['all']['calls']} | {_pc(a['all']['precision'])} | "
                  f"{_pc(a['all']['recall'])} | {_f2(a['all']['f1'])} |"]
    for n in ("v3", "v4"):
        a = acc[f"test|all|{n}|norisk"]
        L += [f"| リスクオフ・オンのラベルを除く | {READER_JA[n]} | {a['n']} | {a['exact']} | {a['all']['calls']} | "
              f"{_pc(a['all']['precision'])} | {_pc(a['all']['recall'])} | {_f2(a['all']['f1'])} |"]
    L += ["", "通貨ごと (新しい半分):", "", "| 通貨 | 正解の向き | v3: 判定 / 正しい / 逆 / F1 | v4: 判定 / 正しい / 逆 / F1 |", "|---|---|---|---|"]
    for cur in CURRENCIES:
        a3, a4 = acc["test|all|v3"]["per"][cur], acc["test|all|v4"]["per"][cur]
        L += [f"| {cur} | {a3['truth']} | {a3['calls']} / {a3['tp']} / {a3['wrong_sign']} / {_f2(a3['f1'])} | "
              f"{a4['calls']} / {a4['tp']} / {a4['wrong_sign']} / {_f2(a4['f1'])} |"]
    L += ["", "見出しごとの判定 (新しい半分):", "", "| 判定 | v3 | v4 |", "|---|---|---|"]
    for k, lab in JUDGE_JA.items():
        L += [f"| {lab} | {t3['judge'].get(k, 0)} | {t4['judge'].get(k, 0)} |"]
    L += [""]
    L += FIXES + [""]
    L += ["### 新しい半分で v3 と v4 の読みが違った見出し", "", "| 見出し | 正解 | v3 | v4 |", "|---|---|---|---|"]
    for r in A["rows"]:
        if r["split"] == "test" and r["v3"] != r["v4"]:
            L += [f"| {_md(r['headline'][:110])} | {_cur(r['truth'])} | {_cur(r['v3'])}{' ✓' if r['v3'] == r['truth'] else ''} | "
                  f"{_cur(r['v4'])}{' ✓' if r['v4'] == r['truth'] else ''} |"]
    L += ["", "### v4 でも残った誤り (新しい半分)", ""] + REMAINING + [""]
    ex = A.get("existing_cases") or {}
    if ex:
        L += ["### 既存のテストの例 (tests/test_news.py、lexicon-v3 向け)", "",
              f"v4 で {ex['pass']} / {ex['cases']} 件が同じ答えになります。" + ("" if not ex["fails"] else " 違うもの: " + "、".join(
                  f"「{f['title']}」(期待 {_cur(f['want'])}、v4 {_cur(_calls(f['v4']))})" for f in ex["fails"]) + "。"), ""]

    # ---- prediction
    L += ["## 結果B: 予測力 (ニュースの信号で次の値動きの向きが当たるか)", ""]
    L += [f"方法は research/news.md と同じです。各時点 (1時間足の終わり) より前に保存済みの見出しだけを使い、本番と同じ集計 "
          "(新しい見出しほど重く、過去48時間、少ないときはゼロに近づける、転載はまとめる) で通貨ごとの圧力を作り、"
          "ペアの信号 = 基準通貨 − 相手通貨 としました。答え合わせは 1・4・24 本後の終値の向き。期間は "
          f"{B['start'][:10]}〜{B['end'][:10]} で、{B['split'][:16]} UTC を境に古い半分と新しい半分に分けました (どちらも"
          "パラメータの調整はしていません)。**的中率** は動きゼロを除いた割合、**平均** は信号の向きに取った平均の動き (bp)、"
          "**t** は日ごとに全ペアの結果を合計した t 値、**相関** は信号の大きさと値動きの相関 (信号がゼロでない時点)。", ""]
    for set_name, R_ in B["sets"].items():
        L += [f"### {SET_JA.get(set_name, set_name)} ({R_['headlines']:,} 本)", "",
              "| 分析 | 点数をつけた見出し | 予測先 | 古い半分: 件数 / 的中率 / 平均 / t / 相関 | 新しい半分: 件数 / 的中率 / 平均 / t / 相関 |",
              "|---|---|---|---|---|"]
        for n in names:
            if n not in R_:
                continue
            ev = R_[n]["eval"]
            for h in H:
                e = ev[h] if h in ev else ev[str(h)]
                row = [f"| {READER_JA[n]} | {R_[n]['scored']:,} | {HOR_JA[h]} "]
                for part, cpart in (("tune", "older"), ("test", "newer")):
                    x = e[part]
                    cc = e["corr"][cpart].get("pearson")
                    row.append(f" {x.get('n', 0):,} / {_pc(x.get('hit'), 1)} / {_sg(x.get('bp'))} / {_sg(x.get('t'))} / {_sg(cc, 3)} ")
                L += ["|".join(row) + "|"]
        L += ["", "信号が「上」を指した割合 (全ペア、ゼロ以外): " + "、".join(
            f"{READER_JA[n]} {_pc(R_[n]['up_share'])}" for n in names if n in R_) + "。過去24時間の値動きとの相関: " + "、".join(
            f"{READER_JA[n]} {_sg(R_[n]['momentum_corr'], 2)}" for n in names if n in R_) + "。", ""]
    L += ["### v4 と v3 の比較 (的中率の差、ポイント)", "",
          "| 見出し | 予測先 | 古い半分 | 新しい半分 |", "|---|---|---|---|"]
    for set_name, R_ in B["sets"].items():
        if "v3" not in R_ or "v4" not in R_:
            continue
        for h in H:
            cells = []
            for part in ("tune", "test"):
                a, b = (_ev(R_[n]["eval"], h)[part].get("hit") for n in ("v3", "v4"))
                cells.append(f"{(b - a) * 100:+.1f}" if a is not None and b is not None else "–")
            L += [f"| {SET_JA.get(set_name, set_name)} | {HOR_JA[h]} | {cells[0]} | {cells[1]} |"]
    cons = consistent(B)
    L += ["", f"両方の半分で同じ向きに t ≥ 2 となった組み合わせ (見出し {len(B['sets'])} 種 × 分析 × 予測先 3): "
          + ("なし。" if not cons else "、".join(cons) + "。"), ""]
    L += PREDICT_NOTE + [""]
    L += ["## 注意点", ""] + CAVEATS + [""]
    L += ["## 本番への提案", ""] + PROPOSAL + [""]
    L += [f"(実行時間 {res['runtime_s']:.0f} 秒。再現: `python -m aifx.research_news_v4` (保存済みのデータだけを使用)。"
          "取得: `python -m aifx.research_news_v4 fetch` (Google ニュース・本番のフィード)、`fetch-doc` (GDELT DOC API)、"
          "`finbert` (FinBERT の点数。torch と transformers が必要)。)"]
    return "\n".join(L) + "\n"


def _ev(ev: dict, h: int) -> dict:
    return ev[h] if h in ev else ev[str(h)]


def consistent(B: dict) -> list[str]:
    """Set / reader / horizon combinations with t >= 2 of the same sign in both halves."""
    out = []
    for set_name, R_ in B["sets"].items():
        for n in ("v3", "v4", "finbert"):
            if n not in R_:
                continue
            for h in H:
                e = _ev(R_[n]["eval"], h)
                a, b = e["tune"].get("t"), e["test"].get("t")
                if a is not None and b is not None and abs(a) >= 2 and abs(b) >= 2 and a * b > 0:
                    out.append(f"{set_name}/{n}/{h}h")
    return out


def _md(s: str) -> str:
    return s.replace("|", "／")


def _cur(d: dict) -> str:
    return " ".join(f"{c}{'+' if v > 0 else '−'}" for c, v in sorted(d.items())) or "なし"


def _predict_summary(res: dict) -> list[str]:
    B = res["b"]["sets"]
    out = []
    g = B.get("gkg", {})
    if "v3" in g and "v4" in g:
        def hits(n, part):
            ev = g[n]["eval"]
            return "・".join(_pc((ev[h] if h in ev else ev[str(h)])[part].get("hit"), 1) for h in H)
        out.append(f"- **予測力は、v3 でも v4 でもありませんでした (予想どおり)。** GDELT の見出しで、1・4・24時間後の的中率は "
                   f"v3 が古い半分 {hits('v3', 'tune')}、新しい半分 {hits('v3', 'test')}、v4 が古い半分 {hits('v4', 'tune')}、"
                   f"新しい半分 {hits('v4', 'test')} でした。読み取りが正しくなっても、見出しの多くは「すでに起きた値動き」や"
                   "「すでに知られた方針」を伝えているため、次の値動きの向きの情報にはなりません。詳しくは結果B。")
    return out


LABEL_RULES = [
    "### ラベル付けの規則 (筆者が見出しだけを読んで判定)",
    "",
    "1. 対象は USD・JPY・EUR・GBP・AUD。見出しから読み取れる「その通貨が上がる (+) / 下がる (−)」を付け、読み取れなければ"
    "ラベルなし。根拠を通貨ごとに記録 (move / policy / data / yields / intervention / risk)。",
    "2. **相場の動き (move)**: 通貨・ペアの上昇・下落 (予想・見通しを含む)。ペアは基準通貨 +・相手通貨 − (USD/JPY 上昇 → USD+ JPY−)、"
    "「A が B に対して上昇」→ A+ B−、「Pound to Euro」「Euro to Dollar」は前の通貨の値。見出しが動きと逆の見通しを述べる場合は"
    "見通しを優先。水準だけ (「158円台」)、横ばい・もみ合い、「steadies」はなし。日本語の「円安・円高」は円だけに付ける。",
    "3. 新興国通貨 (ルピー、ペソ、ナイラ、人民元など) の対ドル相場からは米ドルの向きを付けない。ただし「弱い米指標がドルの重しに」の"
    "ようにドル自体を述べる部分は付ける。主要国通貨どうし (USD/CAD など) は対象通貨の側だけ付ける。",
    "4. **金融政策 (policy)**: 利上げ・利上げ観測・タカ派発言 → +、利下げ・ハト派・利上げ観測の後退 → −。据え置きだけはなし "
    "(「据え置き、利上げに含み」は +)。政治家が利下げを求める、価格・税・予算の「hike」、住宅ローン金利はなし。方向の分からない"
    "疑問形 (「FRB は利上げするか?」) はなし。",
    "5. **経済指標 (data)**: 予想より強い・インフレ加速・雇用増 → +、弱い・インフレ鈍化・雇用の伸び鈍化・成長率見通しの下方修正 → −"
    " (国が見出しから分かる場合)。",
    "6. **金利 (yields)**: 米国債利回りの上昇 → USD+、低下 → USD−。米国以外の利回りは、利上げ観測と結びつく場合だけ policy として付ける。",
    "7. **介入 (intervention)**: 日本の為替介入・介入の観測・口先介入・レートチェック → JPY+ (見出しが円そのものの下落を報じて"
    "いればそちらを優先)。",
    "8. **リスク (risk)**: 広い株式市場 (世界・米国・欧州・アジア・日経・豪州など) がリスク要因 (戦争、緊張、関税、不安など) と"
    "ともに動いた見出し、または「安全資産への逃避」「リスク回避」を明示する見出しだけ。下落 = リスクオフ (JPY+、AUD−、USD+)、"
    "上昇 = リスクオン (JPY−、AUD+)。戦争・政治だけ、原油・金だけ、個別株、新興国の株価指数はなし。ある通貨自体の動きが"
    "見出しにあれば、その通貨はそちらを優先。",
    "9. 通貨と関係のない語 (Dollar Tree、5,000-pound、Sterling Heights、Glen Powell、Sandra Bullock、Board of Elections の BOE、"
    "Aussie = オーストラリア人) はなし。政治の話はなし (通貨や市場との結びつきが書かれていない限り)。",
]

FIXES = [
    "### v4 で直した主な誤りの種類 (古い半分と、同じ期間のラベルなしの見出しを読んで作成)",
    "",
    "1. **リスクオフ・オンの誤判定 (最大の誤り)**: v3 は「war」「tensions」「attack」「crisis」と市場の語 (pound、sterling、"
    "treasurer、dollar など通貨の語を含む) があればリスクオフと数え、「Asian stocks rise as Iran tensions ease」「Nikkei rebounds as"
    " tensions ease」までリスクオフ (円高・豪ドル安) にしていました。v4 は「広い株式市場の動き + リスク要因」か「安全資産・"
    "リスク回避」の明示だけを数え、向きは株の動きから取ります (上昇ならリスクオン)。原油・金だけの見出し、個別株 (「Thermax shares"
    " crash」)、新興国の指数 (Sensex、Kospi)、「3,300-pound car ... crash」「Sterling Heights firefighter ... crash」はなしです。",
    "2. **通貨・中銀と同じ綴りの語**: pound (重さ・動詞)、Sterling (地名・会社名)、BOE (選挙管理委員会)、Bailey・Bullock・Powell "
    "(同姓の別人)、Aussie (オーストラリア人・豪州企業)、「Fed Govt」(ナイジェリア連邦政府)、代名詞の us は、為替・金融政策の文脈が"
    "ある場合だけ通貨・中銀とみなします。",
    "3. **金融政策の手がかり**: 「hikes defence spending」「energy price hikes」「defense budget hike」を利上げに、「mortgage rate "
    "rises」を利上げに、「weighs cutting Fed meetings」を利下げに、「easing geopolitical concerns」を金融緩和に読んでいました。"
    "「hike」は中銀か「rate」がある見出しだけ、「cutting」「easing」は金利・金融の語と組のときだけにしました。"
    "「hike doubts grow」「slash bets on ... hike」「cuts odds of Fed hike」「dims Fed hike bets」「利上げ期待僅かに後退」を逆向きに、"
    "「Will the Fed raise rates?」のような疑問形と「Trump calls for Fed to cut」のような政治家の要求は向きなしにしました。",
    "4. **経済指標**: 「ASML tops forecasts」(企業決算) を米指標の上振れとしていました。上振れ・下振れは指標の語 (雇用、CPI、GDP"
    " など) がある場合だけにし、「US hiring slowed」「inflation drop」「drop in US inflation」「IMF downgrades Australia's growth"
    " forecast」「negative jobs report」を読むようにしました。",
    "5. **通貨ペア**: v3 は USD/JPY 型 (USD・EUR・GBP・AUD 対 JPY・USD) しか読まず、「EUR/AUD bulls」を豪ドル高に、「USD/CNY dips」"
    "を米ドル安にしていました。v4 は主要10通貨のすべての組、「Pound to Euro」「Pound Euro」「Euro to Dollar」の語の組、bullish/"
    "bearish、「recovery in EUR/USD」「positive on the USD」「USD selloff」「persistent weakness」を読み、ペアの片方だけ向きが"
    "分かったときはもう片方を逆向きにします。新興国通貨の組と「dollar ... on Taipei forex market」は主要通貨の情報とみなしません。",
    "6. **動きの動詞**: 「reaches a 40-year low」「stumbles」「struggles」「hits a three-month peak」を追加し、「yen intervention "
    "weakens dollar」の「weakens」のように別の名詞 (intervention、bets、yields など) が主語の動詞は通貨の動きにしません。",
    "7. **米金利**: 「Aussie shares ... as bond yields spike」を米金利の上昇にしていました。米国の文脈がある場合だけ米ドルに付け、"
    "「higher U.S. Treasury yields」「30-year yields to highest」を読むようにしました。",
    "8. **介入**: v3 は見出しに何かの通貨の動きがあると介入を数えず (「Japan could still intervene ... weaker dollar」)、ペア表記の"
    "「USD/JPY: strong suspicion of intervention」も読めませんでした。円そのものの動きがない限り円高要因とし、"
    "「decisive forex action」「ready to act」「断固たる対応」も口先介入として数えます。",
    "9. **日本語**: 全角英数字を半角にそろえ (「ＦＲＢ」「１６０円」)、「ドル/円」「米ドル／円」「ユーロ・ドル」などの表記、ペアの"
    "後ろの最後の向きの語 (「163円目前で急反落」「反発も依然として売りシグナル優勢」「160円割れ」「伸び悩み、上昇分を削る」)、"
    "「ドルの戻りを阻む」「上値が重い」「ユーロ優位」、「政策金利を1%に引き上げ」「企業物価7.1%上昇」「CPIは鈍化」、"
    "「物価上昇『それほど強くない』」の否定を読むようにしました。「円安受け『断固たる対応』」の円安は背景として扱い口先介入を数え、"
    "「交錯」「攻防」のある見出しは政策・介入から向きをつけません。「緩和」は金融緩和だけ (「下押し緩和」「緊張緩和」は除く)、"
    "リスクは英語と同じく株の動きとリスク要因か「リスク回避」の明示だけ (「和平合意」だけではリスクオンにしない)。",
]

REMAINING = [
    "新しい半分で v4 がまだ誤った主な型 (v4 はこの半分を見て直していません。直すなら、次の期間のラベルで確かめてから):",
    "",
    "- 米金利の言い回しの取りこぼし: 「Treasury Yields Extend Highs」「yields continue to break higher」「U.S. 10-Year Yield "
    "Surpasses 5%」「futures rebound as Treasury yields pull back」。",
    "- 動詞・表記の取りこぼし: 「BOJ set to lift rates」(lift = 利上げ)、「AUD/USD Crushed」、「rallying dollar」(形容詞の位置)、"
    "「長期金利…早期利上げ観測で」(日本の話だが円・日銀の語がない)、「Euro: … supports Dollar」のユーロ側。",
    "- 読み違い: 「Dollar Pares Weekly Gains」を「gains」でドル高に、「米ドル対円は下落」を「円は下落」と、「Brent … but Dollar "
    "Refuses to Follow」をドル安に、「Can the S&P 500 Rally as Treasury Yields Rise?」をリスクオンに、「インフレ急上昇から再び"
    "『ソフトランディング』を期待」を利上げ要因に、「利上げは今回でしばらく見送り」を利上げに。",
    "- 文脈・条件の判断が要るもの: 「FRB利上げ観測に債券投資家は懐疑的」(懐疑的 = 観測の否定)、「タカ派演出の植田日銀、本音は"
    "『連続利上げ難しい』」、「Waller … could decide whether there is a rate hike」、「日銀利上げ加速なら」(条件)、"
    "「Fed hikes usually pound stocks」(一般論)。キーワードの規則では限界がある型です。",
]

PREDICT_NOTE = [
    "- 読み取りが良くなっても、予測力は出ませんでした。v4 は v3 よりリスクの誤判定がずっと少なく、信号がいつも同じ向きを指す"
    "偏りも小さくなりましたが、的中率は 50% の前後で、古い半分と新しい半分で t の符号がそろわない (または両方とも小さい) 形は"
    "前回の研究と同じです。",
    "- ニュースの信号と過去24時間の値動きの相関は正で、見出しは主に「すでに起きた値動き」を伝えています。見出しを正しく読むほど"
    "この性質は強くなります (「円急落」を正しく円安と読めば、それは過去の値動きです)。",
    "- 前回の研究と同じく、ニュースは予測の中心を動かさず (BETA_PRIOR = 0)、「いま何が起きているか」の表示に使うのが適切です。"
    "読み取りの改善は、その表示の正しさのためのものです。",
]

CAVEATS = [
    "- ラベルは筆者1人が付けました。リスクオフ・オン (規則8) や見通し記事の向き (規則2) には判断が入ります。リスクのラベルを"
    "除いた成績も表に示しました (結論は同じ)。",
    "- 修正は古い半分と、同じ期間のラベルなしの見出しだけを読んで作り、v4 を固定してから新しい半分を1回だけ採点しました。"
    "ただし、見出しを選ぶ規則とラベルの規則は両方の半分に共通なので、規則そのものの偏りは両方に入ります。",
    "- 見出しは 300 本で、通貨ごとの正解の向きは数十本です (EUR・GBP・AUD は特に少ない)。通貨ごとの数字の誤差は大きいです。",
    "- 予測力の検証は約3か月 (各半分は約6週間) です。的中率 51〜52% の小さな効果は見分けられません。",
    "- Google ニュースの古い記事は日付だけのものが多く、1日遅らせて使ったため、1・4時間後の検証には不利です。",
    "- FinBERT は文の肯定・否定の判定で、通貨への割り当ては「見出しで最初に出てくる通貨に肯定・否定、ほかの通貨には逆」という"
    "単純な規則にしました。英語だけで、日本語の見出しには使っていません。",
]

PROPOSAL = [
    "1. **aifx/news.py の見出し分析を lexicon-v4 に置き換えました** (読み取りの正しさのため。予測は変わりません)。",
    "   - lexicon-v3 はこの比較を再現できるように aifx/news_v3.py にそのまま残しています。呼び出し方と出力の形"
    " ({\"by\", \"cur\", \"men\", \"top\"}) は同じなので、pressures や画面の側の変更はありません。",
    "   - 保存済みの見出しは保存時の分析 (lexicon-v3) を持っているので、過去の予測の再計算 (`aifx audit`) はそのまま一致します。"
    "新しい見出しから v4 の分析が保存されます (\"by\" で区別できます)。",
    "   - tests/test_news.py: 通らなかった1件 (「ドル買い優勢、ドル円は158円に迫る 原油安は円高につながらず」) は、v4 がペアの動き"
    " (158円に迫る = ドル高・円安) を読んで JPY− をつけるためで、テストの目的 (「円高につながらず」を円高と読まない) に合わせて"
    "`got.get(\"JPY\", 0) <= 0` に変えました。v4 で直した読み取りの例もテストに加えています。",
    "2. **ニュースの予測への重み (BETA_PRIOR = 0) は変えない。** 読み取りの改善でも予測力は出ませんでした。",
    "3. **FinBERT などの汎用の感情分析モデルは入れない。** 手元で無料で動きますが、通貨の向きの読み取りは v4 より悪く、"
    "PyTorch などの大きな依存が増えます。",
    "4. **GDELT DOC API には頼らない。** この環境では大半の要求が拒否されました。無料の Google ニュースの RSS (本番の取得元) と"
    "GDELT の公開ファイルで十分です。",
]


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        print(fetch_gnews())
    elif sys.argv[1:] == ["fetch-doc"]:
        print(fetch_doc())
    elif sys.argv[1:] == ["finbert"]:        # needs torch and transformers on the path
        titles = [h["title"] for h in pool() if h["lang"] == "en"] + [r["headline"] for r in load_labels() if r["lang"] == "en"]
        print(len(finbert_scores(titles, threads=3)))
    else:
        run()
