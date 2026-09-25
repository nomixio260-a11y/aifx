"""Can news predict FX direction? GDELT history, the live headline analyzer, leakage-free tests.

Data (all free, analysed locally):
- GDELT GKG 2.1 raw files (data.gdeltproject.org): one line per article with its page title, source,
  tone and the organisations, people and currency themes GDELT found in it, published every 15
  minutes. One batch per hour (a quarter of all articles, the quarter-hour rotating) for 2026-06-26 ..
  2026-09-24; only rows about USD, JPY, EUR, GBP or AUD are kept (data/history/gdelt/gkg/).
- The GDELT DOC 2.0 API was the first choice, but from the research sandbox's shared outgoing IP it
  refused almost every request ("Please limit requests to one every 5 seconds") however far apart they
  were spaced. Every attempt is logged (data/history/gdelt/_attempts.jsonl, at most ``MAX_ATTEMPTS``);
  the few answers are used to cross-check the tone rebuilt from the raw files.
- Hourly prices (history.load_hourly).

Features, per currency and hour, known at the forecast origin (the end of an hourly bar):
(a) GDELT tone and coverage share: the average tone of the articles about the currency in the last k
    hours, its change against the trailing week ("surprise") and the coverage spike.
(b) The live analyzer (news.analyze_lexicon, via an identical copy with switches) on every page title,
    aggregated exactly as news.pressures does (recency weight, 48 h lookback, shrinkage, syndicated
    copies grouped into stories). A headline counts from 10 minutes after its GDELT batch time.
Pair signal = base minus quote. Targets: the sign of the log move 1, 4 and 24 hourly bars after the
origin. First 60 % of origins tune (sign, window, thresholds), last 40 % test. Scores: hit rate, mean
signed move (bp), t statistic from daily sums pooled over the seven pairs, coverage, and selective
accuracy of the strongest 5/10/20 % of signals (thresholds from the tune period).

    python -m aifx.research_news        # writes research/news.md and research/news.json
"""

from __future__ import annotations

import hashlib
import html
import io
import json
import math
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from . import news as N
from .data import CURRENCIES, PAIRS, USER_AGENT

REPORT_DIR = Path("research")
GDELT_DIR = history.HIST_DIR / "gdelt"


# ------------------------------------------------------------ GDELT DOC API

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
# GDELT allows one request every 5 seconds per IP. The sandbox shares its outgoing IP, so requests are
# spaced at least 20 s apart and a refusal is followed by a 30-45 s pause.
MIN_GAP_S = 20.0
BACKOFF_S = 30.0
MAX_ATTEMPTS = 400         # all DOC requests ever sent for this study, refusals included
ATTEMPT_LOG = GDELT_DIR / "_attempts.jsonl"
_last_request = [0.0]
# Hourly tone timelines for 2026-08-01 .. 08-08 (a 7-day window is the longest GDELT returns hourly),
# used to check the tone rebuilt from the raw files.
DOC_QUERIES = {
    "USD": '("Federal Reserve" OR "US dollar" OR greenback OR FOMC OR "Treasury yields")',
    "JPY": '("Bank of Japan" OR yen OR Ueda)',
    "EUR": '(ECB OR euro OR Lagarde OR eurozone)',
    "GBP": '("Bank of England" OR sterling OR "British pound")',
    "AUD": '(RBA OR "Reserve Bank of Australia" OR "Australian dollar")',
}
DOC_WEEK = ("20260801000000", "20260808000000")


def _attempts() -> list[dict]:
    if not ATTEMPT_LOG.exists():
        return []
    return [json.loads(line) for line in ATTEMPT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def _note_attempt(tag: str, status: int, size: int) -> None:
    with ATTEMPT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": tag, "status": status,
                            "bytes": size}) + "\n")


def _cache_name(params: dict, tag: str) -> Path:
    key = urllib.parse.urlencode(sorted(params.items()))
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_\-]+", "_", tag)[:80]
    return GDELT_DIR / f"{slug}_{h}.json"


def cached(params: dict, tag: str) -> dict | None:
    path = _cache_name(params, tag)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("response")
    return None


def gdelt(params: dict, tag: str, log=print, retries: int = 6) -> dict | None:
    """One DOC 2.0 API call, cached on disk; None when it is not cached and could not be fetched."""
    path = _cache_name(params, tag)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("response")
    GDELT_DIR.mkdir(parents=True, exist_ok=True)
    url = GDELT_URL + "?" + urllib.parse.urlencode(params)
    for _ in range(retries):
        if len(_attempts()) >= MAX_ATTEMPTS:
            log(f"  gdelt: request budget ({MAX_ATTEMPTS}) used up")
            return None
        gap = time.monotonic() - _last_request[0]
        if gap < MIN_GAP_S:
            time.sleep(MIN_GAP_S - gap)
        _last_request[0] = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            status = 200
        except urllib.error.HTTPError as exc:
            raw, status = exc.read().decode("utf-8", errors="replace"), exc.code
        except Exception as exc:              # network error: pause and try again
            raw, status = str(exc), 0
        _note_attempt(tag, status, len(raw))
        if status == 200 and "Please limit requests" not in raw[:200]:
            try:
                obj = json.loads(raw, strict=False) if raw.strip() else {}
            except json.JSONDecodeError:
                obj = {"error": raw[:300], "raw": raw}     # a plain-text error: keep it so it is not asked again
            path.write_text(json.dumps({"url": url, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "response": obj}, ensure_ascii=False), encoding="utf-8")
            return obj
        log(f"  gdelt {status} ({raw[:60]!r}); waiting")
        time.sleep(BACKOFF_S + random.uniform(0, 15))
    return None


def doc_tone_request(cur: str) -> tuple[dict, str]:
    tag = "probe_tone_7d" if cur == "JPY" else f"doc_tone_{cur}_7d"
    return ({"query": DOC_QUERIES[cur], "mode": "timelinetone", "startdatetime": DOC_WEEK[0],
             "enddatetime": DOC_WEEK[1], "format": "json"}, tag)


def fetch_doc_checks(log=print) -> dict:
    """Try the DOC tone timelines used for the cross-check (a few polite attempts each)."""
    return {cur: gdelt(*doc_tone_request(cur), log=log) is not None for cur in CURRENCIES}


def attempt_summary() -> dict:
    at = _attempts()
    ok = sum(a["status"] == 200 for a in at)
    return {"attempts": len(at), "answered": ok, "refused": len(at) - ok}


# ------------------------------------------------------------ analyzer variants
#
# A copy of news.analyze_lexicon with switches for the variants tested here.
# With no option set it returns exactly what news.analyze_lexicon returns
# (checked on every headline by ``check_identical``).

VARIANT_OPTS = ("us_case", "neg", "attr", "risk_ctx", "no_risk", "no_em", "ent", "move_cur")
# "ent": names and verbs the live lexicon lacks (the Fed chair in 2026 is Kevin Warsh; "yen eases" is a fall)
EXTRA_EN_ENT = {"USD": [r"\bwarsh\b", r"\bbessent\b", r"\bgreenback'?s?\b"],
                "JPY": [r"\bkatayama\b", r"\bmimura\b"],
                "GBP": [r"\breeves\b"],
                "AUD": [r"\bchalmers\b"]}
_EN_ENT_PLUS = {cur: pats + EXTRA_EN_ENT.get(cur, []) for cur, pats in N.EN_ENT.items()}
EXTRA_UP = r"edges? (?:up|higher)|inch(?:es|ed)? (?:up|higher)|ticks? (?:up|higher)|steadies|recoups|holds gains"
EXTRA_DOWN = (r"eases|eased|softens|softened|edges? (?:down|lower)|inch(?:es|ed)? (?:down|lower)|ticks? (?:down|lower)|"
              r"pares gains|gives up gains|(?:hovers|holds|trades|stays) (?:near|around|at) [\w\- ]{0,15}lows?")
_EN_VERB_PLUS = re.compile(r"\b(?:(?P<up>" + EXTRA_UP + "|" + N.EN_UP + r")|(?P<down>" + EXTRA_DOWN + "|" + N.EN_DOWN + r"))")
# "rate cut odds fall", "hike bets drop": the expectation turned around
_REVERSE_AFTER_PLUS = re.compile(r"[\s\-]*(?:[\w\-]+\s+){0,2}(?:bets?|expectations?|hopes?|odds|pricing|fears?|chances?)\s+"
                                 r"(?:fall|falls|fell|drop|drops|dropped|decline|declines|declined|ease|eases|eased|slip|slips|"
                                 r"shrink|shrinks|diminish|recede|recedes|fade|fades|faded|cool|cools)")


def _entities_en_plus(t: str) -> list[tuple[int, int, str]]:
    ents = []
    for cur, pats in _EN_ENT_PLUS.items():
        for p in pats:
            for m in re.finditer(p, t):
                ents.append((m.start(), m.end(), cur))
    ents.sort(key=lambda e: (e[0], -(e[1] - e[0])))
    out, last_end = [], -1
    for e in ents:
        if e[0] >= last_end:
            out.append(e)
            last_end = e[1]
    return out


_NEG_MOVE = re.compile(r"\b(?:fails? to|failed to|struggles? to|struggled to|unable to|not|no longer|yet to|"
                       r"refuses? to|little|barely|hardly)\b|n't\b")
_NEG_CUE = re.compile(r"(?:\b(?:no|not|without|never|nor)|n't)\s+(?:[\w\-]+\s+){0,2}$")
# "jobs data douse Fed rate hike bets": the verb before the cue turns it around
_REVERSE_BEFORE_PLUS = re.compile(r"\b(?:douse[sd]?|dampen\w*|tempers?|tempered|curb\w*|dash\w*|cool(?:s|ed|ing)?|erode[sd]?)\s+"
                                  r"(?:[\w\-']+\s+){0,3}$")
# "ceasefire with Iran is 'over'", "truce crumbles": a risk-on word undone
_RISK_ON_UNDONE = re.compile(r"(?:[\s\W]+[\w'\-]+){0,4}?[\s\W]+(?:over|crumbl\w*|collaps\w*|ends?|ended|breaks? down|broke down|"
                             r"fails?|failed|falters?|violat\w*|in doubt|jeopardi\w*|shattered|unravel\w*)\b")
# "move_cur": a move verb only moves a currency named as a currency (not "Australian property gains",
# "US payroll gains"), and "A rises against the dollar" moves the dollar the other way
_CUR_WORD = re.compile(r"(?:u\.?s\.? )?dollars?|greenback'?s?|usd|dxy|yen|jpy|euros?|eur|pound|sterling|gbp|cable|aussie|"
                       r"australian dollar|aud")
_AGAINST = re.compile(r"\b(?:against|versus|vs\.?)\s+(?:the\s+)?(?:u\.?s\.?\s+)?(dollar|greenback|yen|euro|pound|sterling|aussie)\b")
_AGAINST_CUR = {"dollar": "USD", "greenback": "USD", "yen": "JPY", "euro": "EUR", "pound": "GBP", "sterling": "GBP", "aussie": "AUD"}
# words that end a clause in an English headline: a policy or data cue is attributed within its clause
_CLAUSE = re.compile(r"[;:|]|,\s|\s[-–—]\s|\b(?:as|after|while|but|despite|amid|ahead of|before|and)\b")
_MARKET_CTX = re.compile(r"\b(?:markets?|stocks?|shares|equities|investors|traders|safe[- ]haven|risk[- ](?:off|on|appetite|"
                         r"sentiment|aversion|assets)|currenc\w*|forex|fx|yields?|bonds?|treasur\w*|dollar|yen|euro|sterling|"
                         r"pound|aussie|gold|oil|wall street|nikkei|s&p|ftse|dow|nasdaq|dax|sensex|nifty|hang seng|asx|"
                         r"kospi|stoxx)\b")
_MARKET_CTX_JA = re.compile(r"株|相場|市場|円|ドル|ユーロ|ポンド|金利|為替|投資家|リスク")


def _clause_bounds(t: str, pos: int) -> tuple[int, int]:
    lo, hi = 0, len(t)
    for m in _CLAUSE.finditer(t):
        if m.end() <= pos:
            lo = m.end()
        elif m.start() > pos:
            hi = m.start()
            break
    return lo, hi


def analyze(title: str, lang: str, default_cur: str | None = None, opts: frozenset = frozenset()) -> dict:
    """news.analyze_lexicon with optional changes (``opts`` from VARIANT_OPTS)."""
    effects: dict[str, float] = {}
    topics: set[str] = set()

    def add(cur, v, topic):
        if cur in CURRENCIES and v:
            effects[cur] = effects.get(cur, 0.0) + v
            topics.add(topic)

    if lang == "ja":
        t = title
        ents = N._entities_ja(t)
        masked = t
        for word, eff in N.JA_MOVES:
            while (i := masked.find(word)) >= 0:
                if not N.JA_NOT_MOVE.match(t, i + len(word)):
                    for cur, v in eff.items():
                        add(cur, v, "fx_move")
                masked = masked[:i] + "＿" * len(word) + masked[i + len(word):]
        single = masked
        for word, _, _ in N.JA_PAIRS:
            single = single.replace(word, "＿" * len(word))
        for m in N.JA_CUR_MOVE.finditer(single):
            cur = N.JA_CUR_CODE[m.group(1)]
            if cur == "USD" and m.start() > 0 and single[m.start() - 1] in "豪米加":
                cur = "USD" if single[m.start() - 1] == "米" else None
            add(cur, 1 if m.group("up") else -1, "fx_move")
        for m in N.JA_YEN_RATE.finditer(single):
            add("JPY", 1 if m.group("up") else -1, "fx_move")
        for word, base, quote in N.JA_PAIRS:
            start = 0
            while (i := t.find(word, start)) >= 0:
                rest = t[i + len(word): i + len(word) + 8]
                d = 1 if N.JA_PAIR_UP.match(rest) else -1 if N.JA_PAIR_DOWN.match(rest) else 0
                if d:
                    add(base, d, "fx_move")
                    add(quote, -d, "fx_move")
                start = i + len(word)
        cues = [(N.JA_HAWK, 0.8, "policy"), (N.JA_DOVE, -0.8, "policy"), (N.JA_BEAT, 0.6, "data"),
                (N.JA_MISS, -0.6, "data"), (N.JA_POLITICS, -0.3, "politics")]
        yield_up, yield_down = N.JA_YIELD_UP, N.JA_YIELD_DOWN
        risk_off, risk_on, intervene, verbal = N.JA_RISK_OFF, N.JA_RISK_ON, N.JA_INTERVENE, N.JA_VERBAL
        market_ctx = _MARKET_CTX_JA

        def reversed_at(m):
            return bool(N.JA_REVERSE.match(t, m.end()))
    else:
        t = title.lower()
        pairs = list(N.EN_PAIR.finditer(t))
        ents_all = _entities_en_plus(t) if "ent" in opts else N._entities_en(t)
        verb = _EN_VERB_PLUS if "ent" in opts else N.EN_VERB
        ents = [e for e in ents_all if not any(p.start() <= e[0] < p.end() for p in pairs)]
        if "us_case" in opts and len(t) == len(title):
            # "us" is only the United States when written "US" or "U.S." (not the pronoun in "tell us")
            ents = [e for e in ents if not (e[2] == "USD" and t[e[0]:e[1]] == "us" and title[e[0]:e[1]] != "US")]
        for s, e, cur in ents:
            if "move_cur" in opts and cur in CURRENCIES and not _CUR_WORD.fullmatch(t[s:e]):
                continue
            d = 0
            m = verb.search(t, e, e + 40)
            if m and m.start() - e <= 25 and not any(e <= s2 < m.start() for s2, _, _ in ents):
                d = 1 if m.group("up") else -1
                if "neg" in opts and _NEG_MOVE.search(t[e:m.start()]):
                    d = 0                                   # "yen fails to rally", "dollar not falling"
            before = t[max(0, s - 14):s]
            if re.search(r"\b(?:stronger|firmer)\s+(?:the\s+)?$", before):
                d = 1
            elif re.search(r"\b(?:weaker|softer)\s+(?:the\s+)?$", before):
                d = -1
            if re.search(N.EN_OBJ_DOWN + r"$", t[max(0, s - 24):s]):
                d = -1
            elif re.search(N.EN_OBJ_UP + r"$", t[max(0, s - 24):s]):
                d = 1
            if not d:
                continue
            if cur == "OTHER":
                if "no_em" in opts:
                    continue                                # "rupee falls against the dollar" says little about majors
                if re.search(r"against (?:the )?(?:u\.?s\.? |us )?dollar", t[e:e + 60]):
                    add("USD", -0.5 * d, "fx_move")
                continue
            add(cur, d, "fx_move")
            if "move_cur" in opts and (a := _AGAINST.search(t, e, e + 70)) and _AGAINST_CUR[a.group(1)] != cur:
                add(_AGAINST_CUR[a.group(1)], -d, "fx_move")
        for p in pairs:
            m = N.EN_PAIR_VERB.match(t, p.end(), p.end() + 40)
            if not m:
                continue
            d = 1 if m.group("up") else -1
            base, quote = p.group(1).upper(), p.group(2).upper()
            add(base, d, "fx_move")
            add(quote, -d, "fx_move")
        cues = [(N.EN_HAWK, 0.8, "policy"), (N.EN_DOVE, -0.8, "policy"), (N.EN_BEAT, 0.6, "data"),
                (N.EN_MISS, -0.6, "data"), (N.EN_POLITICS, -0.4, "politics")]
        yield_up, yield_down = N.EN_YIELD_UP, N.EN_YIELD_DOWN
        risk_off, risk_on, intervene, verbal = N.EN_RISK_OFF, N.EN_RISK_ON, N.EN_INTERVENE, N.EN_VERBAL
        market_ctx = _MARKET_CTX

        def reversed_at(m):
            rev = bool(N.EN_REVERSE_BEFORE.search(t[max(0, m.start() - 40):m.start()]) or N.EN_REVERSE_AFTER.match(t, m.end()))
            if "neg" in opts and not rev and _NEG_CUE.search(t[max(0, m.start() - 30):m.start()]):
                rev = True                                  # "no rate cut", "not hawkish"
            if "neg" in opts and not rev and _REVERSE_AFTER_PLUS.match(t, m.end()):
                rev = True                                  # "rate hike odds fall"
            if "neg" in opts and not rev and _REVERSE_BEFORE_PLUS.search(t[max(0, m.start() - 40):m.start()]):
                rev = True                                  # "data douse rate hike bets"
            return rev

    for pattern, value, topic in cues:
        for m in re.finditer(pattern, t):
            if "attr" in opts and lang != "ja":
                lo, hi = _clause_bounds(t, m.start())
                inside = [x for x in ents if lo <= x[0] < hi]
                cur = N._nearest(inside, m.start()) or N._nearest(ents, m.start()) or default_cur
            else:
                cur = N._nearest(ents, m.start()) or default_cur
            if topic == "policy" and reversed_at(m):
                value_m = -value
            else:
                value_m = value
            add(cur, value_m, topic)
    if re.search(yield_up, t):
        add("USD", 0.6, "yields")
    if re.search(yield_down, t):
        add("USD", -0.6, "yields")
    risk_ok = "no_risk" not in opts and ("risk_ctx" not in opts or bool(market_ctx.search(t)))
    on = re.search(risk_on, t) if risk_ok else None
    undone = bool(on) and "neg" in opts and lang != "ja" and bool(_RISK_ON_UNDONE.match(t, on.end()))
    if risk_ok and (re.search(risk_off, t) or undone):
        for cur, v in N.RISK_OFF_EFFECT.items():
            add(cur, v, "risk")
    if on and not undone:
        for cur, v in N.RISK_ON_EFFECT.items():
            add(cur, v, "risk")
    mentioned = sorted({c for _, _, c in ents if c in CURRENCIES} | ({default_cur} if default_cur else set()))
    if re.search(intervene, t) and "JPY" in mentioned and "fx_move" not in topics:
        add("JPY", 0.8, "intervention")
    elif re.search(verbal, t) and "JPY" in mentioned and "fx_move" not in topics:
        add("JPY", 0.5, "intervention")
    scores = {c: round(math.tanh(v / 1.5), 3) for c, v in sorted(effects.items()) if abs(v) > 1e-9}
    return {"by": N.ANALYZER, "cur": scores, "men": mentioned, "top": sorted(topics)}

# ------------------------------------------------------- GDELT GKG raw files
#
# The DOC API refused almost every request from the shared outgoing IP of the research sandbox, so the
# study is built from GDELT's raw Global Knowledge Graph (GKG 2.1) files instead: static files published
# every 15 minutes (no API, no rate limit), one line per article with its page title, source, tone and the
# organisations it names. One 15-minute batch per hour is downloaded (the quarter-hour rotates), only the
# rows about the five currencies are kept (the raw zip files are not stored).


GKG_URL = "https://data.gdeltproject.org/gdeltv2/{}.gkg.csv.zip"
GKG_DIR = GDELT_DIR / "gkg"
GKG_START = pd.Timestamp("2026-06-26 00:00", tz="UTC")
GKG_END = pd.Timestamp("2026-09-24 10:00", tz="UTC")
_OTHER_DOLLAR = r"(?<!australian )(?<!canadian )(?<!zealand )(?<!hong kong )(?<!singapore )(?<!taiwan )(?<!aussie )"
# per currency: words in the title, or the central bank, its governor or the currency among the
# organisations, people and themes GDELT found in the article text (the ground the DOC queries cover)
GKG_TITLE = {
    "USD": re.compile(_OTHER_DOLLAR + r"\bdollar\b|\bgreenback\b|\bfed\b|federal reserve|\bfomc\b|treasury yields?|\bwarsh\b|\bpowell\b"),
    "JPY": re.compile(r"\byen\b|\bboj\b|bank of japan|\bueda\b"),
    "EUR": re.compile(r"\beuros?\b|\becb\b|\blagarde\b|euro ?zone|euro area"),
    "GBP": re.compile(r"\bsterling\b|\bpound\b|bank of england|\bboe\b|\bbailey\b"),
    "AUD": re.compile(r"australian dollar|\baussie\b|\brba\b|reserve bank of australia|\bbullock\b"),
}
GKG_ORG = {"USD": ("federal reserve", "federal open market committee"), "JPY": ("bank of japan",),
           "EUR": ("european central bank",), "GBP": ("bank of england",), "AUD": ("reserve bank of australia",)}
# GDELT's currency themes mark a mention anywhere in the text ("ECON_WORLDCURRENCIES_DOLLARS", plural, is
# mostly dollar amounts and is left out)
GKG_THEME = {"USD": {"DOLLAR", "US_DOLLAR", "US_DOLLARS", "UNITED_STATES_DOLLAR", "UNITED_STATES_DOLLARS", "GREENBACK"},
             "JPY": {"YEN", "JAPANESE_YEN"},
             "EUR": {"EURO", "EUROS"},
             "GBP": {"BRITISH_POUND", "BRITISH_POUNDS", "POUND_STERLING", "STERLING", "POUND", "POUNDS"},
             "AUD": {"AUSTRALIAN_DOLLAR", "AUSTRALIAN_DOLLARS"}}
GKG_PERSON = {"USD": ("kevin warsh", "jerome powell"), "JPY": ("kazuo ueda",), "EUR": ("christine lagarde",),
              "GBP": ("andrew bailey",), "AUD": ("michele bullock",)}
_TITLE_RE = re.compile(rb"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


def _dt(t: pd.Timestamp) -> str:
    return t.strftime("%Y%m%d%H%M%S")


def gkg_batches(start: pd.Timestamp = GKG_START, end: pd.Timestamp = GKG_END) -> list[pd.Timestamp]:
    """One 15-minute batch per hour; the quarter-hour rotates with the hour and the day."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    return [h + pd.Timedelta(minutes=15 * ((h.hour + h.dayofyear) % 4)) for h in hours]


def parse_gkg(raw: bytes) -> dict:
    """Rows about the five currencies: [title, domain, url, tone, currencies matched, of which in the title]."""
    n, rows = 0, []
    for line in raw.split(b"\n"):
        if not line:
            continue
        n += 1
        f = line.split(b"\t")
        if len(f) < 27:
            continue
        m = _TITLE_RE.search(f[26])
        title = html.unescape(m.group(1).decode("utf-8", "replace")).strip() if m else ""
        orgs = f[13].decode("utf-8", "replace").lower()
        persons = f[11].decode("utf-8", "replace").lower()
        themes = {t[21:] for t in f[7].decode("utf-8", "replace").split(";") if t.startswith("ECON_WORLDCURRENCIES_")}
        tl = title.lower()
        in_title = [c for c in CURRENCIES if GKG_TITLE[c].search(tl)]
        curs = [c for c in CURRENCIES if c in in_title or any(o in orgs for o in GKG_ORG[c])
                or themes & GKG_THEME[c] or any(x in persons for x in GKG_PERSON[c])]
        if not curs or not title:
            continue
        try:
            tone = float(f[15].split(b",")[0])
        except ValueError:
            tone = float("nan")
        rows.append([title[:300], f[3].decode("utf-8", "replace"), f[4].decode("utf-8", "replace")[:300],
                     round(tone, 3), ",".join(curs), ",".join(in_title)])
    return {"n": n, "rows": rows}


def _get(url: str, timeout: float = 60.0) -> tuple[int, bytes]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return 200, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except Exception:
        return 0, b""


GKG_STATS = GKG_DIR / "_downloaded.json"


def _add_downloaded(n: int) -> None:
    st = json.loads(GKG_STATS.read_text()) if GKG_STATS.exists() else {"files": 0, "bytes": 0}
    st["files"] += 1
    st["bytes"] += n
    GKG_STATS.write_text(json.dumps(st))


def fetch_gkg(log=print, start: pd.Timestamp = GKG_START, end: pd.Timestamp = GKG_END) -> dict:
    """Download and reduce every planned batch not cached yet (one file per UTC day)."""
    GKG_DIR.mkdir(parents=True, exist_ok=True)
    batches = gkg_batches(start, end)
    by_day: dict[str, list[pd.Timestamp]] = {}
    for b in batches:
        by_day.setdefault(b.strftime("%Y%m%d"), []).append(b)
    stats = {"files": 0, "bytes": 0, "missing": 0}
    t0 = time.time()
    for day, bs in sorted(by_day.items()):
        path = GKG_DIR / f"{day}.json"
        have = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        todo = [b for b in bs if _dt(b) not in have]
        for b in todo:
            key = _dt(b)
            for attempt in range(3):
                status, raw = _get(GKG_URL.format(key))
                if status in (200, 404):
                    break
                time.sleep(5 * (attempt + 1))
            if status != 200:
                have[key] = {"n": 0, "rows": [], "missing": status}
                stats["missing"] += 1
                continue
            stats["files"] += 1
            stats["bytes"] += len(raw)
            _add_downloaded(len(raw))
            z = zipfile.ZipFile(io.BytesIO(raw))
            have[key] = parse_gkg(z.read(z.namelist()[0]))
            time.sleep(0.2)
        if todo:
            path.write_text(json.dumps(have, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            log(f"gkg {day}: {len(todo)} batches, {sum(len(have[_dt(b)]['rows']) for b in bs)} rows "
                f"({stats['files']} files, {stats['bytes'] / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
    return stats


def load_gkg() -> dict:
    """Cached GKG rows -> headlines (time = batch time) and hourly tone / coverage share per currency.

    Hourly series are indexed by the hour the batch fell in; a batch of hour H is at most 45 minutes
    past H and public ~5 minutes after its time, so at an origin on the hour the newest usable hour is
    the previous one (lag 1)."""
    heads: list[dict] = []
    tone_rows: list[tuple] = []
    n_batches = n_missing = 0
    for path in sorted(GKG_DIR.glob("2*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for key, b in day.items():
            t = pd.Timestamp(key[:8] + " " + key[8:], tz="UTC")
            if b.get("missing"):
                n_missing += 1
                continue
            n_batches += 1
            per = {c: [] for c in CURRENCIES}
            for title, domain, url, tone, curs, in_title in b["rows"]:
                cs = curs.split(",")
                for c in cs:
                    if tone == tone:
                        per[c].append(tone)
                heads.append({"pub": t, "curs": cs, "in_title": in_title, "lang": "en", "url": url,
                              "domain": domain.lower().removeprefix("www."), "title": title})
            tone_rows.append((t.floor("h"), b["n"], {c: (float(np.mean(v)) if v else np.nan, len(v)) for c, v in per.items()}))
    idx = pd.DatetimeIndex([r[0] for r in tone_rows])
    tone = pd.DataFrame({c: [r[2][c][0] for r in tone_rows] for c in CURRENCIES}, index=idx)
    vol = pd.DataFrame({c: [r[2][c][1] / max(r[1], 1) * 100 for r in tone_rows] for c in CURRENCIES}, index=idx)
    count = pd.DataFrame({c: [r[2][c][1] for r in tone_rows] for c in CURRENCIES}, index=idx)
    keep = ~idx.duplicated(keep="last")
    dl = json.loads(GKG_STATS.read_text()) if GKG_STATS.exists() else {}
    return {"tone": tone[keep].sort_index(), "vol": vol[keep].sort_index(), "count": count[keep].sort_index(),
            "heads": heads, "batches": n_batches, "missing": n_missing,
            "articles": int(sum(r[1] for r in tone_rows)), "megabytes": round(dl.get("bytes", 0) / 1e6)}


# ---------------------------------------------------------------- headlines

PUBLISH_DELAY = pd.Timedelta(minutes=10)     # a GKG batch is public a few minutes after its time
FIN_DOMAINS = ("reuters.com", "bloomberg.com", "cnbc.com", "marketwatch.com", "wsj.com", "ft.com", "fxstreet.com",
               "forexlive.com", "investing.com", "finance.yahoo.com", "nasdaq.com", "rttnews.com", "econotimes.com",
               "morningstar.com", "barrons.com", "nikkei.com", "japantimes.co.jp", "kyodonews.net", "afr.com",
               "cityam.com", "businessinsider.com", "forbes.com", "kitco.com", "fxempire.com", "dailyfx.com",
               "actionforex.com", "poundsterlinglive.com", "exchangerates.org.uk", "livemint.com",
               "economictimes.indiatimes.com", "business-standard.com", "japantoday.com", "theglobeandmail.com",
               "fortune.com", "thestreet.com", "benzinga.com", "seekingalpha.com", "zerohedge.com")


def build_items(heads: list[dict], opts: frozenset = frozenset()) -> list[dict]:
    """Headlines as the live system would have stored them: one item per normalised title (the first
    time it was seen), published at its GDELT batch time, stored 10 minutes later, and kept only when the
    analyzer scores it (news.is_relevant)."""
    seen: dict[str, dict] = {}
    for h in sorted(heads, key=lambda x: (x["pub"], x["url"])):
        iid = N.stable_id(N._norm_key(h["title"]))
        if iid in seen:
            continue
        seen[iid] = {"id": iid, "src": "gdelt", "title": h["title"], "lang": h["lang"], "domain": h["domain"],
                     "pub": h["pub"], "published_at": N.iso(h["pub"].to_pydatetime()),
                     "fetched_at": N.iso((h["pub"] + PUBLISH_DELAY).to_pydatetime()),
                     "an": analyze(h["title"], h["lang"], None, opts)}
    return sorted((it for it in seen.values() if it["an"]["cur"]), key=lambda it: (it["published_at"], it["id"]))


def story_ids(items: list[dict], similarity: float | None = N.STORY_SIMILARITY) -> np.ndarray:
    """news.stories over the whole sample (the live system groups within the 48 h it looks at)."""
    if similarity is None:
        return np.arange(len(items))
    old = N.STORY_SIMILARITY
    N.STORY_SIMILARITY = similarity
    try:
        st = N.stories(items)
    finally:
        N.STORY_SIMILARITY = old
    return np.array([st[it["id"]] for it in items])


def pressure_panel(items: list[dict], origins: pd.DatetimeIndex, story: np.ndarray, tau: float = N.TAU_HOURS,
                   look: float = N.LOOKBACK_HOURS, shrink: float = N.SHRINK, weight=None, topics=None) -> np.ndarray:
    """news.pressures at every origin (rows) for each currency (columns, CURRENCIES order)."""
    pub = np.array([it["pub"].value for it in items], dtype=np.int64)
    S = np.zeros((len(items), len(CURRENCIES)))
    M = np.zeros_like(S)
    for i, it in enumerate(items):
        if topics is not None and not (set(it["an"]["top"]) & topics):
            continue
        for cur, v in it["an"]["cur"].items():
            j = CURRENCIES.index(cur)
            S[i, j], M[i, j] = v, 1.0
    w = np.ones(len(items)) if weight is None else np.asarray(weight, float)
    out = np.zeros((len(origins), len(CURRENCIES)))
    o_ns = origins.as_unit("ns").asi8
    lo_all = np.searchsorted(pub, o_ns - int(look * 3.6e12), side="left")
    hi_all = np.searchsorted(pub, o_ns - PUBLISH_DELAY.value, side="left")   # stored strictly before the origin
    for k, (lo, hi) in enumerate(zip(lo_all, hi_all)):
        if hi <= lo:
            continue
        sl = slice(lo, hi)
        _, inv, cnt = np.unique(story[sl], return_inverse=True, return_counts=True)
        age = (o_ns[k] - pub[sl]) / 3.6e12
        d = w[sl] * np.exp(-age / tau) / cnt[inv]
        out[k] = (d @ S[sl]) / (shrink + d @ M[sl])
    return out


def check_identical(heads: list[dict]) -> dict:
    """Our analyzer copy against news.analyze_lexicon (must be identical)."""
    bad = [h["title"] for h in heads if analyze(h["title"], h["lang"]) != N.analyze_lexicon(h["title"], h["lang"])]
    return {"checked": len(heads), "different": len(bad), "examples": bad[:3]}


def check_pressures(items: list[dict], story: np.ndarray, origins: pd.DatetimeIndex) -> float:
    """Largest difference between pressure_panel and news.pressures itself at the given origins."""
    fast = pressure_panel(items, origins, story)
    live_items = [{**it, "src": "gn-en-fx"} for it in items]          # weight 1.0 in news.SOURCE_WEIGHT
    worst = 0.0
    for k, o in enumerate(origins):
        press = N.pressures(live_items, o.to_pydatetime())
        for j, c in enumerate(CURRENCIES):
            worst = max(worst, abs(press[c]["p"] - fast[k, j]))
    return worst


# -------------------------------------------------------------------- prices

H = (1, 4, 24)
TUNE_SHARE = 0.6
SELECT = (0.05, 0.10, 0.20)


def _fwd(c: np.ndarray, h: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    out[:-h] = np.log(c[h:] / c[:-h]) * 1e4
    return out


def price_panel(start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Per pair: origins (end of each hourly bar) in [start, end], forward moves (bp) 1/4/24 bars ahead, the
    time-of-day expectation for the next bar (season.py, as the server computes it), the average forward
    move by origin hour over the year before ``start`` (point in time) and the move over the past 24 hours."""
    from . import season
    out = {}
    for code, pair in PAIRS.items():
        df = history.load_hourly(code)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        c = df["close"].to_numpy(float)
        origin = pd.DatetimeIndex(df.index + pd.Timedelta(hours=1))
        fwd = {h: _fwd(c, h) for h in H}
        past = np.full(len(c), np.nan)
        past[24:] = np.log(c[24:] / c[:-24]) * 1e4
        keep = np.asarray((origin >= start) & (origin <= end))
        o_keep = origin[keep]
        drift = np.zeros(len(o_keep))
        days = o_keep.normalize()
        for day in days.unique():
            st = season.slot_stats(df, day.to_pydatetime(), 60)
            m = np.asarray(days == day)
            drift[m] = season.bar_drift(st, o_keep[m], 60)[0]
        before = np.asarray((origin < start) & (origin >= start - pd.Timedelta(days=365)))
        hr_all = origin.hour.to_numpy()
        hour_mean = {}
        for h in H:
            mu = np.zeros(24)
            for hr in range(24):
                sel = before & (hr_all == hr) & np.isfinite(fwd[h])
                mu[hr] = fwd[h][sel].mean() if sel.sum() > 30 else 0.0
            hour_mean[h] = mu[hr_all[keep]]
        out[code] = {"pair": pair, "origin": o_keep, "fwd": {h: fwd[h][keep] for h in H}, "season": drift,
                     "hour_mean": hour_mean, "past24": past[keep]}
    return out


# ------------------------------------------------------------------- scoring

def score(sig: np.ndarray, fwd: np.ndarray, block: np.ndarray) -> dict:
    """Calls (sig != 0) against the forward move (bp); t from daily sums of the called moves, pairs pooled."""
    m = (sig != 0) & np.isfinite(fwd) & np.isfinite(sig)
    n_all = int(np.isfinite(fwd).sum())
    if m.sum() < 30:
        return {"n": int(m.sum()), "cover": float(m.sum() / max(n_all, 1))}
    s, f, b = np.sign(sig[m]), fwd[m], block[m]
    nz = f != 0
    signed = s * f
    sums = pd.Series(signed).groupby(b).sum()
    sd = sums.std(ddof=1)
    return {"n": int(m.sum()), "cover": float(m.sum() / max(n_all, 1)), "hit": float(np.mean(np.sign(f[nz]) == s[nz])),
            "bp": float(signed.mean()),
            "t": float(sums.mean() / sd * math.sqrt(len(sums))) if sd > 0 and len(sums) > 2 else None}


def stack(P: dict, sigs: dict[str, np.ndarray], h: int, split: pd.Timestamp, demean: bool = False) -> dict:
    """All pairs pooled: signal, target, daily block, and whether each row is in the tune period."""
    s, f, b, tune = [], [], [], []
    for code, p in P.items():
        s.append(sigs[code])
        f.append(p["fwd"][h] - (p["hour_mean"][h] if demean else 0.0))
        b.append(p["origin"].tz_convert(None).normalize().asi8)
        tune.append(np.asarray(p["origin"] < split))
    return {"s": np.concatenate(s), "f": np.concatenate(f), "b": np.concatenate(b), "tune": np.concatenate(tune)}


def evaluate(P: dict, sigs: dict[str, np.ndarray], split: pd.Timestamp, sign: float = 1.0, demean: bool = False) -> dict:
    """sign(signal) scored on tune and test for each horizon, plus selective accuracy: the strongest
    5/10/20 % of signals, with the |signal| thresholds taken from the tune period."""
    out = {}
    for h in H:
        d = stack(P, sigs, h, split, demean)
        s = d["s"] * sign
        f_tune = np.where(d["tune"], d["f"], np.nan)
        f_test = np.where(~d["tune"], d["f"], np.nan)
        r = {"tune": score(s, f_tune, d["b"]), "test": score(s, f_test, d["b"])}
        a = np.abs(s)
        tune_nz = a[d["tune"] & (a > 0) & np.isfinite(a) & np.isfinite(d["f"])]
        sel = {}
        for q in SELECT:
            if len(tune_nz) < 100:
                break
            thr = float(np.quantile(tune_nz, 1 - q))
            strong = a >= thr
            sel[f"{int(q * 100)}%"] = {"thr": thr, "tune": score(np.where(strong, s, 0), f_tune, d["b"]),
                                       "test": score(np.where(strong, s, 0), f_test, d["b"])}
        r["select"] = sel
        out[h] = r
    return out


def _mean_t(ev: dict, part: str) -> float:
    ts = [ev[h][part].get("t") for h in H if ev[h][part].get("t") is not None]
    return float(np.mean(ts)) if ts else float("nan")


def _mean_hit(ev: dict, part: str) -> float:
    hs = [ev[h][part].get("hit") for h in H if ev[h][part].get("hit") is not None]
    return float(np.mean(hs)) if hs else float("nan")


# ------------------------------------------------------------ GDELT tone and volume

TONE_K = (3, 12, 48)          # hours averaged
TRAIL = 168                   # the trailing week a "surprise" or a spike is measured against


def tone_features(G: dict, origins: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Per feature: an array (origins x CURRENCIES) of what was known at each origin (see load_gkg: the
    newest usable hour is the one before the origin's).

    tone_k: average tone of the articles about the currency in the last k hours; surprise_k: tone_k minus
    the trailing week's; volume_k: log of the currency's share of all articles in the last k hours over
    the trailing week's (a coverage spike)."""
    tone, vol, count = G["tone"], G["vol"], G["count"]
    grid = pd.date_range(tone.index.min(), tone.index.max(), freq="h")
    pos = grid.get_indexer(origins - pd.Timedelta(hours=1))
    ok = pos >= 0

    def at(series: pd.Series) -> np.ndarray:
        v = np.full(len(origins), np.nan)
        v[ok] = series.to_numpy(float)[pos[ok]]
        return v

    tone_k: dict[int, dict[str, pd.Series]] = {k: {} for k in TONE_K + (TRAIL,)}
    vol_k: dict[int, dict[str, pd.Series]] = {k: {} for k in TONE_K + (TRAIL,)}
    for c in CURRENCIES:
        tn = tone[c].reindex(grid)
        vl = vol[c].reindex(grid)                            # NaN where the batch is missing
        w = count[c].reindex(grid).fillna(0.0).where(tn.notna(), 0.0)
        tw = tn.fillna(0.0) * w
        for k in tone_k:
            den = w.rolling(k, min_periods=1).sum()
            tone_k[k][c] = tw.rolling(k, min_periods=1).sum() / den.where(den > 0)
            vol_k[k][c] = vl.rolling(k, min_periods=max(1, k // 2)).mean()
    out: dict[str, np.ndarray] = {}
    for k in TONE_K:
        out[f"tone_{k}"] = np.column_stack([at(tone_k[k][c]) for c in CURRENCIES])
        out[f"surprise_{k}"] = np.column_stack([at(tone_k[k][c] - tone_k[TRAIL][c]) for c in CURRENCIES])
        out[f"volume_{k}"] = np.column_stack([at(np.log((vol_k[k][c] + 1e-3) / (vol_k[TRAIL][c] + 1e-3)))
                                              for c in CURRENCIES])
    return out


def pair_sigs(P: dict, per_cur: np.ndarray, origins: pd.DatetimeIndex, clip: bool = True) -> dict[str, np.ndarray]:
    """Base minus quote for every pair, aligned to each pair's own origins (0 where unknown)."""
    s = pd.DataFrame(per_cur, index=origins, columns=CURRENCIES)
    out = {}
    for code, p in P.items():
        v = s.reindex(p["origin"])
        x = (v[p["pair"].base] - v[p["pair"].quote]).to_numpy(float)
        x = np.where(np.isfinite(x), x, 0.0)
        out[code] = np.clip(x, -1, 1) if clip else x
    return out


# ---------------------------------------------------------------------- study

LEX_VARIANTS = {
    # name: (analyzer options, aggregation settings)
    "base": ((), {}),
    "us_case": (("us_case",), {}),
    "neg": (("neg",), {}),
    "attr": (("attr",), {}),
    "risk_ctx": (("risk_ctx",), {}),
    "no_risk": (("no_risk",), {}),
    "no_em": (("no_em",), {}),
    "ent": (("ent",), {}),
    "move_cur": (("move_cur",), {}),
    "all_fixes": (("us_case", "neg", "attr", "risk_ctx", "no_em", "ent", "move_cur"), {}),
    "tau3": ((), {"tau": 3.0}),
    "tau6": ((), {"tau": 6.0}),
    "tau24": ((), {"tau": 24.0}),
    "look24": ((), {"look": 24.0}),
    "dedup_none": ((), {"similarity": None}),
    "dedup_strict": ((), {"similarity": 0.4}),
    "src_weight": ((), {"src": 2.0}),
    "title_fx": ((), {"title_fx": True}),
    "surprise": ((), {"demean": 168}),
    "moves_only": ((), {"topics": {"fx_move"}}),
    "no_moves": ((), {"topics": {"policy", "data", "yields", "risk", "intervention", "politics"}}),
}
LEX_NAMES = {
    "base": "現在の分析 (本番と同じ)",
    "us_case": "小文字の \"us\" (代名詞) を米国とみなさない",
    "neg": "否定の処理 (\"fails to rally\" を上昇としない、\"no rate cut\"・\"hike odds fall\" を逆向きに)",
    "attr": "金融政策・指標の手がかりを、同じ文節にある通貨に優先して帰属",
    "risk_ctx": "リスクオフ語 (war、tensions など) は市場の文脈があるときだけ数える",
    "no_risk": "リスクオフ・オンの規則を使わない",
    "no_em": "新興国通貨の「対ドルで上昇・下落」からドルの動きを推定しない",
    "ent": "人名・動詞の追加 (Warsh、Bessent、片山、\"eases\"、\"edges lower\" など)",
    "move_cur": "「上昇・下落」は通貨の名前にだけ付ける (\"Australian property gains\" は豪ドル高でない)、\"A rises against the dollar\" はドル安も",
    "all_fixes": "上の7つの修正をすべて (no_risk 以外)",
    "tau3": "時間減衰 3時間 (現在12時間)",
    "tau6": "時間減衰 6時間",
    "tau24": "時間減衰 24時間",
    "look24": "過去24時間の見出しだけ (現在48時間)",
    "dedup_none": "同じ記事の転載をまとめない",
    "dedup_strict": "転載のまとめ方を広く (類似度0.4以上、現在0.6)",
    "src_weight": "金融・為替の専門媒体の重みを2倍",
    "title_fx": "見出しに通貨・中銀の名前がある記事だけ",
    "surprise": "ニュースの圧力から過去1週間の平均を引く (いつも同じ向きの偏りを除き、変化だけを見る)",
    "moves_only": "相場の動きを報じた部分 (fx_move) だけ",
    "no_moves": "相場の動き以外 (金融政策・指標・金利・リスク・介入) だけ",
}


def lexicon_signals(G: dict, P: dict, origins: pd.DatetimeIndex, name: str, cache: dict) -> tuple[dict, list[dict], np.ndarray]:
    opts, agg = LEX_VARIANTS[name]
    key = tuple(sorted(opts))
    if key not in cache:
        cache[key] = build_items(G["heads"], frozenset(opts))
    items = cache[key]
    if agg.get("title_fx"):
        items = [it for it in items if it["an"]["men"]]
    sim = agg.get("similarity", N.STORY_SIMILARITY)
    skey = ("story", key, sim, bool(agg.get("title_fx")))
    if skey not in cache:
        cache[skey] = story_ids(items, sim)
    weight = None
    if "src" in agg:
        weight = np.array([agg["src"] if any(it["domain"] == d or it["domain"].endswith("." + d) for d in FIN_DOMAINS)
                           else 1.0 for it in items])
    press = pressure_panel(items, origins, cache[skey], tau=agg.get("tau", N.TAU_HOURS),
                           look=agg.get("look", N.LOOKBACK_HOURS), weight=weight, topics=agg.get("topics"))
    if "demean" in agg:          # origins are consecutive hours, so the trailing rows are the trailing hours
        df = pd.DataFrame(press)
        press = (df - df.rolling(agg["demean"], min_periods=24).mean()).fillna(0.0).to_numpy()
    return pair_sigs(P, press, origins), items, press


def study(log=print) -> dict:
    t0 = time.time()
    G = load_gkg()
    if not G["heads"]:
        raise RuntimeError("no GKG data cached: run fetch_gkg() first")
    start = GKG_START + pd.Timedelta(hours=72)             # warm-up for the 48 h lookback
    end = GKG_END
    P = price_panel(start, end)
    all_o = np.unique(np.concatenate([p["origin"][np.isfinite(p["fwd"][1])].as_unit("ns").asi8 for p in P.values()]))
    split = pd.Timestamp(all_o[int(len(all_o) * TUNE_SHARE)], tz="UTC")
    origins = pd.date_range(start, end, freq="h")
    log(f"GKG: {G['batches']} batches ({G['missing']} missing), {len(G['heads'])} rows; "
        f"origins {start} .. {end}, split {split}")
    res: dict = {"start": str(start), "end": str(end), "split": str(split),
                 "gkg": {"batches": G["batches"], "missing": G["missing"], "articles": G["articles"],
                         "rows": len(G["heads"]), "megabytes_downloaded": G.get("megabytes")},
                 "doc_api": attempt_summary()}
    res["identical"] = check_identical(G["heads"][::3])
    log(f"analyzer copy identical: {res['identical']['different']} different of {res['identical']['checked']}")

    # --- the live analyzer and its variants
    cache: dict = {}
    lex: dict = {}
    for name in LEX_VARIANTS:
        sigs, items, press = lexicon_signals(G, P, origins, name, cache)
        ev = evaluate(P, sigs, split)
        lex[name] = {"eval": ev, "items": len(items)}
        if name == "base":
            base_sigs, base_items, base_press = sigs, items, press
            lex[name]["eval_flipped"] = evaluate(P, sigs, split, sign=-1.0)
            lex[name]["eval_demeaned"] = evaluate(P, sigs, split, demean=True)
            story = cache[("story", (), N.STORY_SIMILARITY, False)]
            res["pressure_check"] = check_pressures(items, story, origins[200::211][:8])
            log(f"pressure_panel vs news.pressures: max difference {res['pressure_check']:.2e}")
        log(f"lexicon {name:13s} tune t {_mean_t(ev, 'tune'):+.2f} hit {_mean_hit(ev, 'tune'):.3f} | "
            f"test t {_mean_t(ev, 'test'):+.2f} hit {_mean_hit(ev, 'test'):.3f}  ({time.time() - t0:.0f}s)")
    res["lexicon"] = lex
    res["diag"] = diagnostics(G, base_items, split, cache[tuple(sorted(LEX_VARIANTS["all_fixes"][0]))])
    res["diag"]["pressure_nonzero"] = {c: float(np.mean(np.abs(base_press[:, j]) > 1e-9)) for j, c in enumerate(CURRENCIES)}
    res["momentum_corr"] = momentum_corr(P, base_sigs)
    res["pairs"] = pair_breakdown(P, base_sigs, split)

    # --- GDELT tone and coverage
    feats = tone_features(G, origins)
    grid = {}
    for fname, arr in feats.items():
        sigs = pair_sigs(P, arr, origins, clip=False)
        grid[fname] = {"+": evaluate(P, sigs, split), "-": evaluate(P, sigs, split, sign=-1.0)}
    chosen = {}
    for fam in ("tone", "surprise", "volume"):
        for h in H:
            best = max(((f, sg) for f in grid if f.startswith(fam + "_") for sg in "+-"),
                       key=lambda x: grid[x[0]][x[1]][h]["tune"].get("t") or -9.0)
            r = grid[best[0]][best[1]][h]
            sigs = pair_sigs(P, feats[best[0]], origins, clip=False)
            chosen[f"{fam}|{h}"] = {"feature": best[0], "sign": best[1], **r,
                                    "demeaned": evaluate(P, sigs, split, sign=1.0 if best[1] == "+" else -1.0, demean=True)[h]}
    res["tone_grid"] = {f: {sg: {str(h): {k: v for k, v in grid[f][sg][h].items() if k != "select"} for h in H}
                            for sg in "+-"} for f in grid}
    res["tone_chosen"] = chosen
    res["doc_check"] = doc_crosscheck(G)
    res["ja_coverage"] = ja_coverage(log)
    log(f"tone grid done ({time.time() - t0:.0f}s); DOC cross-check {res['doc_check']}")

    # --- on top of the time-of-day effect
    res["season"] = season_interplay(P, base_sigs, split)
    res["tests"] = {"lexicon_variants": len(LEX_VARIANTS), "tone_features": len(grid), "tone_signs": 2, "horizons": len(H),
                    "selective_levels": len(SELECT)}
    res["runtime_s"] = round(time.time() - t0, 1)
    return res


def momentum_corr(P: dict, sigs: dict) -> float:
    """Correlation of the news signal with the pair's own move over the past 24 hours: headlines that
    report a move ("yen falls") repeat what prices already did."""
    x = np.concatenate([sigs[c] for c in P])
    y = np.concatenate([P[c]["past24"] for c in P])
    m = np.isfinite(y) & (x != 0)
    return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 10 else float("nan")


def season_interplay(P: dict, news_sigs: dict, split: pd.Timestamp) -> dict:
    """Does news add to the time-of-day call (season.py) for the next hour?"""
    d = pd.concat([pd.DataFrame({"season": np.sign(p["season"]), "news": np.sign(news_sigs[c]), "f": p["fwd"][1],
                                 "tune": np.asarray(p["origin"] < split), "pair": c,
                                 "day": p["origin"].tz_convert(None).normalize().asi8}) for c, p in P.items()])
    d = d[np.isfinite(d["f"])]
    out = {}
    for part, m in (("tune", d["tune"]), ("test", ~d["tune"])):
        x = d[m & (d["season"] != 0)]
        r = {"season_only": score(x["season"].to_numpy(), x["f"].to_numpy(), x["day"].to_numpy())}
        for lab, mm in (("agree", x["news"] == x["season"]), ("disagree", (x["news"] != 0) & (x["news"] != x["season"]))):
            y = x[mm]
            r[lab] = score(y["season"].to_numpy(), y["f"].to_numpy(), y["day"].to_numpy())
        r["news_in_season_hours"] = score(x["news"].to_numpy(), x["f"].to_numpy(), x["day"].to_numpy())
        y = d[m & (d["season"] == 0)]
        r["news_other_hours"] = score(y["news"].to_numpy(), y["f"].to_numpy(), y["day"].to_numpy())
        # the same comparison within each pair and direction of the time-of-day call (the news signal
        # of some pairs is almost always on one side, so "agree" would otherwise just mean "a down call"),
        # weighted by the number of calls
        x = x[(x["news"] != 0) & (x["f"] != 0)].assign(hit=lambda z: np.sign(z["f"]) == z["season"],
                                                       agree=lambda z: z["news"] == z["season"])
        diffs, weights = [], []
        for _, g in x.groupby(["pair", "season"]):
            a, b = g[g["agree"]], g[~g["agree"]]
            if len(a) >= 10 and len(b) >= 10:
                diffs.append(a["hit"].mean() - b["hit"].mean())
                weights.append(len(g))
        r["agree_minus_disagree_within_pair"] = float(np.average(diffs, weights=weights)) if diffs else None
        r["groups_compared"] = len(diffs)
        r["calls_compared"] = int(np.sum(weights)) if weights else 0
        out[part] = r
    return out


def pair_breakdown(P: dict, sigs: dict, split: pd.Timestamp) -> dict:
    """Per pair: the share of origins where the news signal pointed up, the 24-hour hit rate, and the share
    of 24-hour moves that were up (a signal that always says "down" hits exactly the share of down moves)."""
    out = {}
    for c, p in P.items():
        tune = np.asarray(p["origin"] < split)
        s = sigs[c]
        row = {"up_share": float(np.mean(s[s != 0] > 0)) if (s != 0).any() else None}
        for part, m in (("tune", tune), ("test", ~tune)):
            f = p["fwd"][24]
            k = m & (s != 0) & np.isfinite(f) & (f != 0)
            row[f"hit24_{part}"] = float(np.mean(np.sign(f[k]) == np.sign(s[k]))) if k.sum() > 30 else None
            ff = f[m & np.isfinite(f) & (f != 0)]
            row[f"up24_{part}"] = float(np.mean(ff > 0)) if len(ff) else None      # share of 24 h moves that were up
        out[c] = row
    return out


def doc_crosscheck(G: dict) -> dict:
    """Correlation of the tone rebuilt from the GKG sample with GDELT's own hourly tone timeline (DOC API,
    the week of 2026-08-01, full text and all articles) for the currencies whose timeline was answered."""
    out = {}
    for cur in CURRENCIES:
        resp = cached(*doc_tone_request(cur))
        if not resp or not resp.get("timeline"):
            continue
        s = pd.Series({pd.Timestamp(x["date"].replace("T", " ").replace("Z", ""), tz="UTC"): float(x["value"])
                       for x in resp["timeline"][0]["data"]})
        # a DOC bin is labelled by the end of its hour; the GKG hour H holds a batch inside (H, H + 1 h]
        ours = G["tone"][cur].copy()
        ours.index = ours.index + pd.Timedelta(hours=1)
        both = pd.concat([s, ours], axis=1, keys=["doc", "gkg"], sort=True).dropna()
        daily = both.resample("D").mean().dropna()
        out[cur] = {"hours": len(both), "corr_hourly": round(float(both.corr().iloc[0, 1]), 3),
                    "corr_6h": round(float(both.rolling(6).mean().dropna().corr().iloc[0, 1]), 3),
                    "corr_daily": round(float(daily.corr().iloc[0, 1]), 3) if len(daily) > 3 else None}
    return out


# ---------------------------------------------------------------- diagnostics

SAMPLE_SEED = 7
SAMPLE_SEED_TEST = 11
SAMPLE_N = 30
# My reading of 30 scored headlines drawn from the tune period: the tracked currencies whose value each
# headline points up (+1) or down (-1); {} = nothing about the five currencies' direction.
MANUAL_LABELS: dict[str, dict[str, int]] = {
    'Gold rate today slips below $4,000 for the first time since 2025, silver price also declines. Is a bigger crash coming?': {},
    'Cotality Report: Australian property resale gains hit record $377,000 | The Advocate - Hepburn': {},
    'US payroll employment gains cools in June - London Business News': {'USD': -1},
    'Asian markets choppy as US jobs data douse Fed rate hike bets': {'USD': -1},
    "Oil prices rise 7%, and Dow drops 600 points after Trump says ceasefire with Iran is 'over'": {'JPY': 1, 'AUD': -1, 'USD': 1},
    'Mortgage Rates Rise as Iran Ceasefire Crumbles': {},
    'Donald Trump fires bipartisan federal election commission members – NBC Connecticut': {},
    'U.S. Dollar Climbs Against Most Majors': {'USD': 1},
    'Gold drops as US-Iran strikes revive inflation, rate-hike risks': {'USD': 1},
    'Oil prices spike on fresh US-Iran attacks, tech weighs on stocks again': {'JPY': 1, 'AUD': -1, 'USD': 1},
    'FxWirePro:    AUD/USD firms slightly, but downward resumption looks likely': {'AUD': -1, 'USD': 1},
    "Will Warsh Break the Fed's 56-Year Rate-Hike Streak?": {},
    'Hegseth estimates Iran war has cost $48.5b so far': {},
    'Indian stock market crash: Sensex slides 715 points | Indiablooms - First Portal on Digital News Management': {},
    'USD/JPY at 40-year high as Middle East tensions and hawkish Fed bets drive breakout': {'USD': 1, 'JPY': -1},
    'FTSE 100 falls as Middle East war intensifies': {'JPY': 1, 'AUD': -1, 'USD': 1},
    "DXN's SE Asia Expansion Boosts Aussie Business": {},
    "Fed Didn't Raise Rates After All — Will Mortgage Rates Fall? | National": {'USD': -1},
    'US Fed dissenters call for rate hikes over sustained inflation': {'USD': 1},
    'World stocks are mixed as yen jumps against the dollar, while oil prices slip': {'JPY': 1, 'USD': -1},
    'Gold extends gains on lower oil and softer dollar, markets await US jobs data': {'USD': -1},
    '"War\'s got to end pretty soon": Trump says Iran \'can\'t go much longer,\' expresses optimism over Strait of Hormuz negotiations': {},
    'Stocks gain as weak NFP sees traders pare Fed rate hike bets - Newsquawk US Market Wrap': {'USD': -1},
    'US: Rate expectations declined – Shafaqna English | International Shia News & Fatwas': {'USD': -1},
    "Lucy Powell vows to tackle 'Neets' crisis as thousands to get no exam results": {},
    'Market Watch: AI growth supports market resilience despite oil tensions': {'JPY': -1, 'AUD': 1},
    'Rupee falls 7 paise to 95.40 against US dollar amid FII selling, geopolitical risks': {},
    'Foreign investors sell Japanese stocks for second week on yen intervention concerns': {'JPY': 1},
    'Oil edges higher as Middle East tensions linger': {'JPY': 1, 'AUD': -1, 'USD': 1},
    'Canadian Stocks Poised For Gains As Tariff Pause Signals Trade Deal': {'JPY': -1, 'AUD': 1},
}


MANUAL_LABELS_TEST: dict[str, dict[str, int]] = {
    'Gold advances on softer dollar, stable yields - London Business News': {'USD': -1},
    'The Price Of Diesel Hits 7 Dollars A Gallon In California And A Massive Trade War Just Erupted Between The U.S. And Canada': {},
    "Iran's currency hits record low as US moves to unveil new sanctions": {},
    "Trump's popularity hits record low amid Iran war, finds Reuters Ipsos Poll": {},
    'US sanctions, AI jitters and investors search for safer ground - ATV Today': {'JPY': 1, 'AUD': -1, 'USD': 1},
    'Wall Street Points To Gains As US Raises Pressure On Iran And Trade War With Canada Intensifies': {'JPY': -1, 'AUD': 1},
    'Canada imposes retaliatory tariffs on US goods from September 8 as Trump trade war escalates': {},
    'Canada hits US with retaliatory tariffs as trade war intensifies': {},
    'Rupee falls 13 paise to 95.56 against U.S. dollar in early trade': {},
    'The horrific reason the UK is facing a fresh immigration crisis exposed': {},
    "Fed Governor Chris Waller Just Significantly Upped the Stakes for the Sept. 11 Inflation Report. It Could Decide Whether There is a Rate Hike at the Fed's Upcoming Meeting": {},
    "'Small potatoes': US President Trump downplays war on Iran | US-Israel war on Iran News": {},
    'Stocks dented by inflation risk from rising oil, dicey geopolitics': {'JPY': 1, 'AUD': -1, 'USD': 1},
    'Iran Wants To Extend Enforcement Mechanism To Its Oil-Loading Terminals As The Costs Of War Hit Home - Analysis': {},
    'Dow drops 500 points as oil nears $100 amid Iran war': {'JPY': 1, 'AUD': -1, 'USD': 1},
    'Rupee falls to 95.11 vs US dollar as West Asia tensions drive oil fears': {},
    'Treasury yields spike after $6 billion buyback plan announced': {'USD': 1},
    'Pound to Dollar Forecast: GBP Rate Retreats after Strong US PPI': {'GBP': -1, 'USD': 1},
    'U.S. Dollar Moves Lower As Bessent Boosts Bond Buybacks: Analysis For EUR/USD, GBP/USD, USD/CAD, USD/JPY': {'USD': -1},
    'Global Tensions and High Prices Squash Consumer Outlook Ahead of Fed Decision': {},
    "Why The S&P 500 Isn't Panicking As Oil Surges, War Spreads, The Fed Hikes": {'USD': 1},
    'Trump sees three new GOP defections in House vote to end Iran war': {},
    'Saudis pound Yemen, Houthis fire at Saudi, as Middle East war spreads': {},
    'Heartland economic professor speaks out after Federal Reserve hikes interest rates': {'USD': 1},
    'Pound to Euro Falls after BoE Holds Rates in 6-3 Vote': {'GBP': -1, 'EUR': 1},
    'Economic Watch: Bank of England holds interest rate at 3.75 pct despite energy price hikes': {},
    'Seoul stocks rise 1.65% on chip gains despite Middle East tensions': {'JPY': -1, 'AUD': 1},
    'GLOBAL MARKETS-Tech Rally Boosts Asian Stocks, Dollar Firms on Rate-Hike Wagers': {'USD': 1},
    'Term deposit yield up for fifth week on hawkish Fed signals': {'USD': 1},
    'Aussie shares sink back into red as bond yields spike': {},
}


def _judge(pred: dict[str, float], truth: dict[str, int]) -> str:
    p = {c: int(np.sign(v)) for c, v in pred.items() if v}
    if p == truth:
        return "ok"
    if not truth:
        return "should_be_neutral"
    if any(c in p and p[c] != s for c, s in truth.items()):
        return "wrong_sign"
    if any(c not in truth for c in p):
        return "wrong_currency"
    return "missed"


def diagnostics(G: dict, items: list[dict], split: pd.Timestamp, items_fixed: list[dict] | None = None) -> dict:
    heads = G["heads"]
    uniq = {N._norm_key(h["title"]) for h in heads}
    out: dict = {"rows": len(heads), "unique_titles": len(uniq), "scored_items": len(items),
                 "nonneutral_share": len(items) / max(len(uniq), 1)}
    out["rows_per_currency"] = dict(Counter(c for h in heads for c in h["curs"]))
    out["title_mentions_share"] = float(np.mean([bool(h["in_title"]) for h in heads]))
    out["top_domains"] = Counter(h["domain"] for h in heads).most_common(12)
    out["scored_per_currency"] = dict(Counter(c for it in items for c in it["an"]["cur"]))
    out["topics"] = dict(Counter(tp for it in items for tp in it["an"]["top"]))
    sub = heads[::7]
    out["nonneutral_variants"] = {name: float(np.mean([bool(analyze(h["title"], "en", None, frozenset(opts))["cur"]) for h in sub]))
                                  for name, (opts, _agg) in LEX_VARIANTS.items() if name == "base" or opts}
    # scored tune-period headlines, read by hand
    tune_scored = [it for it in items if it["pub"] < split]
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = sorted(rng.choice(len(tune_scored), size=min(SAMPLE_N, len(tune_scored)), replace=False))
    all_opts = frozenset(LEX_VARIANTS["all_fixes"][0])
    sample = []
    for i in pick:
        it = tune_scored[int(i)]
        row = {"title": it["title"], "domain": it["domain"], "base": it["an"]["cur"],
               "fixed": analyze(it["title"], "en", None, all_opts)["cur"]}
        if it["title"] in MANUAL_LABELS:
            row["truth"] = MANUAL_LABELS[it["title"]]
            row["base_judge"] = _judge(row["base"], row["truth"])
            row["fixed_judge"] = _judge(row["fixed"], row["truth"])
        sample.append(row)
    out["sample"] = sample
    out.update(_sample_summary(sample, "sample"))
    by_opt = {}
    # a fresh sample from the test period, of headlines scored by either analyzer, read after the fixes were
    # written: the out-of-sample check of the fixes
    if items_fixed is not None:
        pool = {it["id"]: it for it in items + items_fixed if it["pub"] >= split}
        pool_l = sorted(pool.values(), key=lambda it: (it["published_at"], it["id"]))
        rng = np.random.default_rng(SAMPLE_SEED_TEST)
        pick = sorted(rng.choice(len(pool_l), size=min(SAMPLE_N, len(pool_l)), replace=False))
        sample_t = []
        for i in pick:
            it = pool_l[int(i)]
            row = {"title": it["title"], "domain": it["domain"], "base": analyze(it["title"], "en")["cur"],
                   "fixed": analyze(it["title"], "en", None, all_opts)["cur"]}
            if it["title"] in MANUAL_LABELS_TEST:
                row["truth"] = MANUAL_LABELS_TEST[it["title"]]
                row["base_judge"] = _judge(row["base"], row["truth"])
                row["fixed_judge"] = _judge(row["fixed"], row["truth"])
            sample_t.append(row)
        out["sample_test"] = sample_t
        out.update(_sample_summary(sample_t, "sample_test"))
    # correct readings with each fix alone (tune sample / test sample)
    for opt in VARIANT_OPTS:
        by_opt[opt] = [sum(_judge(analyze(r["title"], "en", None, frozenset([opt]))["cur"], r["truth"]) == "ok"
                           for r in smp if "truth" in r) for smp in (sample, out.get("sample_test", []))]
    by_opt["all_fixes"] = [sum(r.get("fixed_judge") == "ok" for r in smp) for smp in (sample, out.get("sample_test", []))]
    by_opt["base"] = [sum(r.get("base_judge") == "ok" for r in smp) for smp in (sample, out.get("sample_test", []))]
    out["sample_ok_by_option"] = by_opt
    return out


def _sample_summary(sample: list[dict], name: str) -> dict:
    judged = [r for r in sample if "truth" in r]
    if not judged:
        return {}
    errs: dict[str, Counter] = {"base": Counter(), "fixed": Counter()}
    for r in judged:
        for k in errs:
            for c in set(r[k]) | set(r["truth"]):
                if int(np.sign(r[k].get(c, 0))) != r["truth"].get(c, 0):
                    errs[k][c] += 1
    return {f"{name}_summary": {"n": len(judged), "base": dict(Counter(r["base_judge"] for r in judged)),
                                "fixed": dict(Counter(r["fixed_judge"] for r in judged))},
            f"{name}_errors_per_currency": {k: dict(v) for k, v in errs.items()}}


JA_BATCHES = ("20260714013000", "20260805030000", "20260902050000", "20260917004500", "20260722061500", "20260827234500")
JA_CACHE = GDELT_DIR / "ja_coverage.json"
_JA_LOOSE = re.compile(r"円|ドル|ユーロ|ポンド|為替|日銀|FRB|利上げ|利下げ|金利")
# "円" alone is mostly a price ("11万4800円"); these are about the currency or monetary policy
_JA_FX = re.compile(r"円相場|円安|円高|円買い|円売り|ドル円|ドル高|ドル安|為替|日銀|FRB|利上げ|利下げ|ユーロ|ポンド|豪ドル")


def ja_coverage(log=print) -> dict:
    """How many Japanese articles GDELT's translated stream carries (a few batches, Tokyo daytime)."""
    if JA_CACHE.exists():
        out = json.loads(JA_CACHE.read_text(encoding="utf-8"))
        out["japanese_fx"] = sum(bool(_JA_FX.search(t)) for t in out["titles"])
        return out
    out = {"batches": 0, "megabytes": 0.0, "articles": 0, "japanese": 0, "japanese_fx": 0, "titles": []}
    for key in JA_BATCHES:
        status, raw = _get(f"https://data.gdeltproject.org/gdeltv2/{key}.translation.gkg.csv.zip")
        if status != 200:
            continue
        out["batches"] += 1
        out["megabytes"] += len(raw) / 1e6
        z = zipfile.ZipFile(io.BytesIO(raw))
        for line in z.read(z.namelist()[0]).split(b"\n"):
            f = line.split(b"\t")
            if len(f) < 27:
                continue
            out["articles"] += 1
            if b"srclc:jpn" not in f[25]:
                continue
            out["japanese"] += 1
            m = _TITLE_RE.search(f[26])
            title = html.unescape(m.group(1).decode("utf-8", "replace")) if m else ""
            if _JA_LOOSE.search(title):
                out["titles"].append(title[:120])
        time.sleep(0.5)
    JA_CACHE.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    out["japanese_fx"] = sum(bool(_JA_FX.search(t)) for t in out["titles"])
    log(f"Japanese in GDELT: {out['japanese']} of {out['articles']} articles, {out['japanese_fx']} about FX or policy")
    return out


# -------------------------------------------------------------------- report

def _p(x, d=1):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f}%"


def _t(x):
    return "–" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.2f}"


def _bp(x):
    return "–" if x is None else f"{x:+.2f}"


def _row(label: str, r: dict) -> str:
    return (f"| {label} | {r.get('n', 0):,} | {_p(r.get('cover'), 0)} | {_p(r.get('hit'))} | {_bp(r.get('bp'))} | "
            f"{_t(r.get('t'))} |")


HOR = {1: "1時間後", 4: "4時間後", 24: "24時間後"}
FEAT = {"tone": "トーンの水準", "surprise": "トーンの変化 (過去1週間の平均との差)", "volume": "報道量の急増"}


def report(res: dict) -> str:
    lex = res["lexicon"]
    base = lex["base"]["eval"]
    dg = res["diag"]
    L = ["# ニュースで為替の方向は当たるか (GDELT のニュース、2026年6月〜9月)", ""]
    L += CONCLUSION + [""]

    # ---------------- data
    g = res["gkg"]
    da = res["doc_api"]
    L += ["## データ", "",
          f"- **ニュース**: GDELT (世界中のニュースサイトを15分ごとに集めて公開している無料のデータ) の GKG というファイルから、"
          f"1時間に1回分 (15分ぶん、全記事の約4分の1) を {res['start'][:10]} 〜 {res['end'][:10]} の期間で取り出しました。"
          f"読み込んだ記事 {g['articles']:,} 本のうち、5つの通貨 (米ドル・円・ユーロ・ポンド・豪ドル) に関係する記事 {g['rows']:,} 本"
          f" (見出しの重複を除くと {dg['unique_titles']:,} 本) を使いました。ダウンロードは約 {g['megabytes_downloaded'] / 1000:.1f} GB、"
          f"欠けていた回は {g['missing']} 回です。英語の記事だけです。",
          f"- 「通貨に関係する記事」は、見出しに通貨や中央銀行の名前がある記事か、記事の本文で GDELT がその通貨・中央銀行・総裁を見つけた記事です"
          f" (見出しに名前がある記事は {_p(dg['title_mentions_share'], 0)})。",
          f"- 最初は GDELT の検索 API (DOC API) を使う予定でしたが、この環境の出口の IP アドレスは他の利用者と共有されていて、"
          f"間隔を十分にあけても「5秒に1回まで」という制限で断られました ({da['attempts']} 回試して応答は {da['answered']} 回)。"
          f"そのため、制限のない公開ファイル (GKG) に切り替えました。得られた API の応答は、GKG から作ったトーンの確認に使いました。",
          "- **価格**: 7つの通貨ペアの1時間足 (Yahoo Finance)。", ""]
    dc = res.get("doc_check") or {}
    if dc:
        parts = [f"{c} {v['corr_hourly']:+.2f} / {v['corr_6h']:+.2f} / {_t(v['corr_daily'])}" for c, v in dc.items()]
        L += ["GKG から作ったトーンと、GDELT 自身の API のトーン (2026-08-01〜08-08、全記事・本文検索) の相関 "
              "(1時間ごと / 6時間平均 / 1日平均): " + "、".join(parts) + "。", ""]
    ja = res.get("ja_coverage") or {}
    if ja:
        L += [f"日本語の見出しは使えませんでした。GDELT の翻訳版のファイル (英語以外の記事) を東京の昼間を中心に {ja['batches']} 回分"
              f" ({ja['megabytes']:.0f} MB) 調べたところ、記事 {ja['articles']:,} 本のうち日本語は {ja['japanese']} 本、"
              f"そのうち為替・金融政策の見出しは {ja['japanese_fx']} 本でした。GDELT には日本語のニュースがほとんど入っていません。", ""]

    # ---------------- method
    L += ["## 方法", "",
          "- **予測の時点**: 各1時間足の終わり。その時点より前に公開されていた記事だけを使います"
          " (GDELT の15分ごとのまとまりの時刻の10分後から使えるとみなしました)。",
          "- **見出しの分析**: 本番の aifx/news.py の分析 (キーワードで各通貨への影響を -1〜+1 で採点) をそのまま使い、"
          "本番と同じ集計 (新しい記事ほど重く (12時間で約0.37倍)、過去48時間、記事が少ないときはゼロに近づける、同じ記事の転載はまとめる) で"
          "通貨ごとの「ニュースの圧力」を計算しました。本番と同じ結果になることを確かめています"
          f" (分析: {res['identical']['checked']:,} 本で差 {res['identical']['different']} 本、集計: 最大の差 {res['pressure_check']:.0e})。",
          "- **GDELT のトーン**: 記事ごとのトーン (文章の肯定・否定の度合い) の平均と、全記事に占める割合 (報道量) を通貨ごとに1時間ごとに計算し、"
          f"過去 {'/'.join(str(k) for k in TONE_K)} 時間の水準、過去1週間の平均との差 (変化)、報道量の急増を特徴にしました。",
          "- **通貨ペアの信号** = 基準通貨の値 − 相手通貨の値 (例: 米ドル/円 = 米ドル − 円)。正なら上、負なら下と予想します。",
          f"- **答え合わせ**: 1・4・24本後 (1時間足) の終値が予測の時点より上か下か。前半60% ({res['start'][:10]} 〜 {res['split'][:10]}) で"
          f"設定 (向き・期間・しきい値) を選び、後半40% ({res['split'][:10]} 〜 {res['end'][:10]}) で確かめました。",
          "- **的中率** は動きがゼロの回を除いた割合、**平均** は予想した向きへの平均の動き (bp = 0.01%)、"
          "**t** は日ごとに全ペアの結果を合計して計算した t 値 (2以上で偶然では説明しにくい水準)、**予想した割合** は信号がゼロでない時点の割合です。", ""]

    # ---------------- lexicon results
    L += ["## 結果1: 本番の見出し分析", "",
          "| 予測先 | 期間 | 件数 | 予想した割合 | 的中率 | 平均 (bp) | t |", "|---|---|---|---|---|---|---|"]
    for h in H:
        for part, lab in (("tune", "調整"), ("test", "検証")):
            L.append(_row(f"{HOR[h]} | {lab}", base[h][part]))
    fl = lex["base"]["eval_flipped"]
    dm = lex["base"]["eval_demeaned"]
    L += ["", "時間帯ごとの平均的な動き (研究開始前の1年間で測定) を差し引いた動きで採点すると:", "",
          "| 予測先 | 期間 | 件数 | 予想した割合 | 的中率 | 平均 (bp) | t |", "|---|---|---|---|---|---|---|"]
    for h in H:
        for part, lab in (("tune", "調整"), ("test", "検証")):
            L.append(_row(f"{HOR[h]} | {lab}", dm[h][part]))
    L += ["", "通貨ペアごとの、信号が「上」を指した割合と、24時間後の的中率、その期間に24時間後の値が上がっていた割合"
          " (いつも「下」を指す信号の的中率は、下がっていた割合 = 100% − 上がっていた割合 になります):", "",
          "| 通貨ペア | 信号が「上」を指した割合 | 調整: 的中率 / 上がっていた割合 | 検証: 的中率 / 上がっていた割合 |", "|---|---|---|---|"]
    for c, v in res["pairs"].items():
        L.append(f"| {PAIRS[c].label} | {_p(v['up_share'], 0)} | {_p(v['hit24_tune'])} / {_p(v['up24_tune'])} | "
                 f"{_p(v['hit24_test'])} / {_p(v['up24_test'])} |")
    L += ["", f"逆張り (ニュースと逆の向き) にした場合の t は、調整期間 {', '.join(_t(fl[h]['tune'].get('t')) for h in H)}、"
          f"検証期間 {', '.join(_t(fl[h]['test'].get('t')) for h in H)} (1・4・24時間後) です。"
          f"ニュースの信号と過去24時間の値動きの相関は {res['momentum_corr']:+.2f} で、見出しの多くは「すでに起きた値動き」を伝えています。", ""]
    L += ["### 信号が強いときだけに絞った場合", "",
          "しきい値は調整期間の信号の強さの上位5/10/20%で決め、検証期間にそのまま当てはめました。", "",
          "| 予測先 | 上位 | 調整: 件数 / 的中率 / t | 検証: 件数 | 的中率 | 平均 (bp) | t |", "|---|---|---|---|---|---|---|"]
    for h in H:
        for q, r in base[h]["select"].items():
            tu, te = r["tune"], r["test"]
            L.append(f"| {HOR[h]} | {q} | {tu.get('n', 0):,} / {_p(tu.get('hit'))} / {_t(tu.get('t'))} | {te.get('n', 0):,} | "
                     f"{_p(te.get('hit'))} | {_bp(te.get('bp'))} | {_t(te.get('t'))} |")
    L.append("")

    # ---------------- tone
    L += ["## 結果2: GDELT のトーンと報道量", "",
          "それぞれ、期間 (3・12・48時間) と向き (+ = トーンが高い通貨が上がる、− = 逆) を調整期間の t で選び、検証期間で確かめました。", "",
          "| 特徴 | 予測先 | 選んだ設定 | 調整: 的中率 / t | 検証: 件数 | 予想した割合 | 的中率 | 平均 (bp) | t | 検証 上位10%: 的中率 / t |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in res["tone_chosen"].items():
        fam, h = key.split("|")
        h = int(h)
        tu, te = r["tune"], r["test"]
        s10 = r.get("select", {}).get("10%", {}).get("test", {})
        L.append(f"| {FEAT[fam]} | {HOR[h]} | {r['feature'].split('_')[1]}時間、{r['sign']} | {_p(tu.get('hit'))} / {_t(tu.get('t'))} | "
                 f"{te.get('n', 0):,} | {_p(te.get('cover'), 0)} | {_p(te.get('hit'))} | {_bp(te.get('bp'))} | {_t(te.get('t'))} | "
                 f"{_p(s10.get('hit'))} / {_t(s10.get('t'))} |")
    n_grid = len(res["tone_grid"]) * 2 * len(H)
    L += ["", f"調整期間で比べた組み合わせは {n_grid} 通り (特徴9 × 向き2 × 予測先3) です。", ""]

    # ---------------- season
    L += ["## 結果3: 時間帯の偏りに上乗せできるか", "",
          "本番で使っている時間帯の偏り (aifx/season.py、次の1時間) が方向を示す時間に、ニュースの向きが一致した回と逆の回で、"
          "時間帯の偏りの的中率を比べました。ニュースに情報があれば、一致した回の方がよく当たるはずです。", "",
          "| 期間 | 区分 | 件数 | 的中率 | 平均 (bp) | t |", "|---|---|---|---|---|---|"]
    for part, lab in (("tune", "調整"), ("test", "検証")):
        s = res["season"][part]
        for k, name in (("season_only", "時間帯の偏りのみ (全体)"), ("agree", "ニュースが同じ向き"), ("disagree", "ニュースが逆の向き"),
                        ("news_in_season_hours", "ニュースの向き (偏りのある時間)"), ("news_other_hours", "ニュースの向き (それ以外の時間)")):
            r = s[k]
            L.append(f"| {lab} | {name} | {r.get('n', 0):,} | {_p(r.get('hit'))} | {_bp(r.get('bp'))} | {_t(r.get('t'))} |")
    st, se = res["season"]["tune"], res["season"]["test"]
    L += ["", "全体では「同じ向き」の方がよく当たって見えますが、ニュースの信号はペアごとにほぼ同じ向きなので、"
          "「同じ向き」はペアや呼び方の向き (下げの呼び方が多い) の違いを含みます。同じ通貨ペア・同じ向きの呼び方の中で比べると、"
          f"一致した回の的中率の差は 調整期間 {100 * st['agree_minus_disagree_within_pair']:+.1f} ポイント ({st['calls_compared']:,} 回)、"
          f"検証期間 {100 * se['agree_minus_disagree_within_pair']:+.1f} ポイント ({se['calls_compared']:,} 回) で、上乗せの効果は見られません。"
          "時間帯の偏りを差し引いた動きで採点した結果1の2つ目の表も同じ結論です。", ""]

    # ---------------- variants
    L += ["## 結果4: 見出し分析の改良案", "",
          "1つずつ変えて、調整期間と検証期間の両方で比べました (t と的中率は1・4・24時間後の平均)。"
          "「両方で有効」は、平均の t が調整・検証の両方でプラスで、しかも現在の分析より高いものです"
          " (現在の分析は調整期間でマイナスなので、「マイナスが小さくなった」だけでは改善と数えません)。", "",
          "| 案 | 内容 | 採点した見出し | 調整: 的中率 / t | 検証: 的中率 / t | 検証 (1 / 4 / 24時間後の t) | 両方で有効 |",
          "|---|---|---|---|---|---|---|"]
    bt, bs = _mean_t(base, "tune"), _mean_t(base, "test")
    for name, v in lex.items():
        e = v["eval"]
        mt, ms = _mean_t(e, "tune"), _mean_t(e, "test")
        both = "–" if name == "base" else ("○" if min(mt, ms) > 0 and mt > bt and ms > bs else "×")
        L.append(f"| {name} | {LEX_NAMES[name]} | {v['items']:,} | {_p(_mean_hit(e, 'tune'))} / {_t(mt)} | "
                 f"{_p(_mean_hit(e, 'test'))} / {_t(ms)} | {' / '.join(_t(e[h]['test'].get('t')) for h in H)} | {both} |")
    L += ["", VARIANT_NOTE, ""]

    # ---------------- diagnostics
    L += ["## 結果5: 見出し分析の診断", "",
          f"- 通貨に関係する記事の見出し {dg['unique_titles']:,} 本のうち、分析が何らかの点数をつけたのは {dg['scored_items']:,} 本 "
          f"({_p(dg['nonneutral_share'])}) です。本番の Google News の見出し (為替の記事が中心) より、一般のニュースが多く含まれます。",
          "- 点数をつけた通貨の内訳: " + "、".join(f"{c} {dg['scored_per_currency'].get(c, 0):,}" for c in CURRENCIES) + " 本。",
          "- 点数の理由 (延べ): " + "、".join(f"{TOPIC_JA.get(k, k)} {v:,}" for k, v in sorted(dg["topics"].items(), key=lambda x: -x[1])) + "。",
          "- 改良案ごとの「点数をつけた割合」: " + "、".join(f"{k} {_p(v)}" for k, v in dg["nonneutral_variants"].items()) + "。", ""]
    for key, intro in (("sample", "調整期間の、現在の分析が点数をつけた見出しから30本を無作為に選び、1本ずつ読んで確かめました。"
                                   "改良案 (7つの修正) はこの30本などを読んで作ったので、修正後の成績はこの30本では有利に出ます。"),
                       ("sample_test", "修正を作った後に、検証期間の見出し (現在の分析か修正後のどちらかが点数をつけたもの) から"
                                       "別の30本を選んで確かめました。修正の効果の公平な確認はこちらです。")):
        ss = dg.get(f"{key}_summary")
        if not ss:
            continue
        L += [intro + " 正解は見出しから読み取れる通貨の向きで、為替や市場と関係のない見出しは「点数なし」が正解です。"
              "市場のリスク回避・選好を伝える見出し (株安・原油高と戦争など) は、分析の決まり (リスク回避なら円高・豪ドル安・ドル高) どおりを正解としました。", "",
              "| 判定 | 現在の分析 | 7つの修正後 |", "|---|---|---|"]
        for k, name in JUDGE_JA.items():
            L.append(f"| {name} | {ss['base'].get(k, 0)} | {ss['fixed'].get(k, 0)} |")
        errs = dg.get(f"{key}_errors_per_currency", {})
        L += ["", "通貨ごとの誤り (延べ、現在 → 修正後): " + "、".join(
            f"{c} {errs.get('base', {}).get(c, 0)} → {errs.get('fixed', {}).get(c, 0)}" for c in CURRENCIES) + "。", "",
            "| 見出し | 現在の分析 | 修正後 | 正解 | 判定 (現在 / 修正後) |", "|---|---|---|---|---|"]
        for r in dg[key]:
            if "truth" not in r:
                continue
            L.append(f"| {r['title'][:95].replace('|', '/')} | {_cur(r['base'])} | {_cur(r['fixed'])} | {_cur(r['truth'])} | "
                     f"{JUDGE_JA[r['base_judge']]} / {JUDGE_JA[r['fixed_judge']]} |")
        L.append("")
    ok = dg.get("sample_ok_by_option")
    if ok:
        L += ["修正を1つずつ入れたときに正しく読めた本数 (調整の30本 / 検証の30本):", "",
              "| 案 | 調整 | 検証 |", "|---|---|---|"]
        for k, (a, b) in ok.items():
            L.append(f"| {k} | {a} | {b} |")
        L.append("")
    L += [SAMPLE_NOTE, ""]

    # ---------------- caveats + proposals
    n_var = len(LEX_VARIANTS)
    L += ["## 注意点", "",
          f"- 試した数が多いこと: 見出し分析 {n_var} 通り × 予測先3 = {n_var * 3} 通り、トーン {n_grid} 通り (検証は選んだ9通り)、"
          "強い信号だけの3段階。これだけ試すと、効果がなくても検証期間で t が2を超えるものが偶然いくつか出ます。",
          "- 期間が短いこと: 約3か月 (検証期間は約5週間) です。ニュースの信号は数日続くことが多く、実質的な独立の標本はもっと少なく、"
          "小さな効果 (的中率 51〜52%) は見分けられません。",
          "- GDELT の見出しは全記事の約4分の1で、英語だけです。本番の Google News の見出し (日本語を含む) とは記事の種類が違います。",
          "- 4時間後・24時間後は予測の期間が重なるため、件数ほどの情報はありません (t は日ごとにまとめて計算しています)。", ""]
    L += PROPOSAL + [""]
    L += [f"(実行時間 {res['runtime_s']:.0f} 秒。再現: `python -m aifx.research_news`。データは data/history/gdelt/ にあります。)", ""]
    return "\n".join(L)


TOPIC_JA = {"fx_move": "相場の動き", "risk": "リスクオフ・オン", "policy": "金融政策", "data": "経済指標", "yields": "米金利",
            "intervention": "介入", "politics": "政治"}
JUDGE_JA = {"ok": "正しい", "should_be_neutral": "点数をつけるべきでない", "wrong_sign": "向きが逆", "wrong_currency": "通貨が違う・余分",
            "missed": "通貨の見落とし"}


def _cur(d: dict) -> str:
    return " ".join(f"{c}{'+' if v > 0 else '−'}" for c, v in d.items() if v) or "なし"


# Written after reading the results (research/news.json); the tables are generated from the results.
CONCLUSION = [
    "## 結論",
    "",
    "- **ニュースで為替の方向は当てられませんでした。** 本番の見出し分析 (aifx/news.py と同じ計算) の信号は、調整期間 (6/29〜8/20) では"
    "1・4・24時間後の的中率が 47%・46%・41% と「逆に」当たり、検証期間 (8/20〜9/24) では 49%・51%・54% と当たる側に回りました。"
    "期間によって向きが入れ替わる信号は、予測には使えません。",
    "- **理由は、信号がほぼいつも同じ向きを指していたことです。** 豪ドル/円では信号が「上」を指したことは一度もなく、豪ドル/米ドルは1%、"
    "ポンド/円は5%でした。2026年は中東の戦争の記事が絶えず、\"war\" \"tensions\" などの語がある見出しを「リスク回避 (円高・豪ドル安・ドル高)」と"
    "採点する規則が、点数の理由で最も多かったためです。いつも同じ向きの信号の的中率は、その期間の相場の流れで決まるだけです"
    " (調整期間は7ペアすべてで「24時間後に上がっていた回」が54〜64%と多く、「下」ばかりの信号は外れました。"
    "検証期間はユーロ/米ドルとポンド/米ドルが下がる回が多く (上がった回は35〜37%)、当たる側に回りました)。過去1週間の平均を引いて「変化」だけを見ても、同じように期間で向きが入れ替わりました。",
    "- **GDELT のトーン (記事の肯定・否定の度合い) と報道量も効果なし。** 調整期間で最もよかった設定は、検証期間ではどれも的中率 48〜51%、t はマイナスでした。",
    "- **強い信号だけに絞っても同じです。** 上位5・10・20% に絞ると、調整期間では的中率 25〜45% (逆向き)、検証期間では約50〜61% と、ここでも向きが入れ替わりました。",
    "- **時間帯の偏りへの上乗せもなし。** 本番で使っている唯一の有効な方向 (時間帯の偏り) に、ニュースの向きを重ねても、同じ通貨ペア・同じ向きの中で比べると的中率は上がりませんでした。",
    "- **見出し分析の改良案 (20通り) も、予測は良くしませんでした。** どの案も「調整期間でマイナス、検証期間でプラス」という同じ形のままです。",
    "- **見出しの読み取りは改良できます。** 修正を作った後に検証期間の見出し30本を読んで確かめると、正しく読めたのは現在の分析で 9本、"
    "修正後で 18本でした。効いたのは主に「リスク回避の語は市場の話の見出しでだけ数える」修正です (単独で 9本 → 16本)。ただし読み取りが正しくなっても、予測は良くなりませんでした。",
    "- 期間は約3か月 (検証期間は約5週間)、英語の記事の約4分の1だけです。的中率 51〜52% 程度の小さな効果があっても、この長さでは見分けられません。"
    "「効果がない」とは「この期間とデータでは偶然と区別できる効果が見つからなかった」という意味です。",
]
VARIANT_NOTE = ("どの案も、調整期間ではマイナス、検証期間ではプラスという同じ形で、案による違いは信号の偏りの強さの違いにすぎません。"
                "改良案は20通り、予測先3つで60の比較をしているので、検証期間だけで t が2を超える組み合わせがいくつか出るのは偶然でも起こります。"
                "改良案の一部 (人名の追加や \"eases\" など) は、検証期間を含む数十本の見出しを見て思いついたものです。")
SAMPLE_NOTE = ("見つかった誤りの種類: (1) **為替と関係のない見出しに点数をつける** のが最も多い誤りでした (調整 11本、検証 16本)。"
               "戦争・政治・事件の見出しに \"war\" \"crisis\" \"tensions\" \"attack\" があると、市場の話でなくても「リスク回避」と採点します。"
               "人名や一般の語の取り違え (英国の政治家 Lucy Powell を FRB、\"Aussie business\" や \"Australian property ... gains\" を豪ドル高、"
               "\"Trump's popularity hits record low\" をドル安)、インド・ルピーの対ドル相場 (主要通貨にはほとんど関係がない) もあります。"
               "(2) **向きの誤り**: \"US payroll employment gains cools\" (雇用の伸びの鈍化をドル高に)、\"jobs data douse Fed rate hike bets\" "
               "(利上げ観測の後退をタカ派に)、\"Fed Didn't Raise Rates\" (否定を読まない)、\"ceasefire ... is 'over'\" (停戦の終わりをリスク選好に)、"
               "\"Pound to Euro Falls\" (ポンド/ユーロの下落をユーロ安に)、\"energy price hikes\" (エネルギー価格の上昇を英中銀の利上げに)。"
               "(3) 通貨ごとでは、誤りは米ドル・円・豪ドルに集中しています (リスク回避の規則が円と豪ドルに、人名や \"US\" が米ドルに付くため)。"
               "ユーロとポンドは点数をつけた見出し自体が少なく (ユーロ 187本、ポンド 97本)、ユーロ/米ドルとポンド/米ドルの信号はほとんど米ドル側だけで決まっています。")
PROPOSAL = [
    "## 本番への提案",
    "",
    "1. **ニュースで予測の中心を動かさない。** 現在、予測の中心はニュースの信号で少し傾けられています (aifx/learning.py の BETA_PRIOR = 0.05、"
    "実績に応じて学習)。今回、記事の時刻を守った過去3か月の検証で、ニュースの信号は期間によって向きが入れ替わり、方向の情報がないと分かりました。"
    "BETA_PRIOR を 0 にし、実績で効果がはっきりしたときだけ傾くようにすることを勧めます。ニュースは「何が起きているか」の表示として残せば十分です。",
    "2. **見出し分析の修正 (表示の正確さのため。予測の改善は期待しない)**。移す価値が高い順に:",
    "   - **risk_ctx** (リスク回避・選好の語は、株・債券・為替・原油・金などの市場の語がある見出しでだけ数える): 誤りの最大の原因を減らします。"
    "30本の確認で、正しく読めた本数が 調整 11 → 16、検証 9 → 16。",
    "   - **no_em** (新興国通貨の「対ドルで上昇・下落」から米ドルの動きを推定しない): 検証 9 → 10。効果は小さいものの、「ルピー安 = ドル高」のような主要通貨と関係の薄い推定を止めるだけで、害はありません。",
    "   - **neg** (\"didn't\"、\"douse/dampen ... bets\"、\"hike odds fall\"、\"ceasefire ... over\" の向きを正す) と **move_cur**"
    " (「上昇・下落」は通貨の名前にだけ付け、\"A rises against the dollar\" では米ドルを逆向きに): 調整期間の見出しでは誤りを直しましたが (11 → 14、11 → 12)、"
    "検証の30本には該当する見出しがなく、効果は確認できていません。入れるなら、tests/test_news.py に今回の見出しを例として足してから。",
    "   - **ent** (人名の更新): 2026年の FRB 議長は Warsh 氏で、\"powell\" は別人 (英国の政治家、町の新聞名) に当たるようになっています。"
    "効果は確認できませんでしたが、中央銀行の総裁や財務相が替わったら辞書を更新する、という保守の作業として必要です。",
    "   - 移さなくてよいもの: us_case と attr (見出しの読み取りも予測も変化なし)、時間減衰・過去の期間・転載のまとめ方・媒体の重みの変更 (一貫した効果なし)。",
    "   - 日本語の見出しの分析は、GDELT に日本語の記事がほとんどないため検証できませんでした。今のまま残します。",
    "3. **GDELT のトーンをサーバーに加えることは勧めません。** 理由: (a) トーンも報道量も、検証期間で方向の情報がありませんでした。"
    "(b) 検索 API は「5秒に1回まで」の制限があり、出口の IP を共有する環境 (GitHub Actions など) では今回のように大半が断られる可能性があります"
    " (43 回中 9 回しか応答なし)。(c) 制限のない GKG ファイルは1時間に1回分でも1日約100 MB の取得が必要で、ポンドと豪ドルは記事が少なく、"
    "GDELT 自身のトーンとの相関もほぼゼロでした。無料で手元で計算でき、5通貨で1時間に5回 (5秒に1回より十分少ない) の取得で済むので技術的には可能ですが、"
    "予測に役立たないものを増やす理由はありません。",
]


def run(log=print) -> dict:
    res = study(log=log)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "news.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (REPORT_DIR / "news.md").write_text(report(res), encoding="utf-8")
    return res


if __name__ == "__main__":
    run()
