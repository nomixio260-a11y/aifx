"""News and economic-calendar collection, headline analysis, and news signals.

Collection: Google News searches (Japanese and English), central-bank press
feeds (Fed, ECB, BOJ, BoE, RBA), an FX news feed, and the weekly economic
calendar. Only the headline, publisher, link and times are kept.

Analysis: a transparent bilingual keyword model, run locally with no paid
service, scores each headline's likely effect on USD, JPY, EUR, GBP and AUD
(-1 weaker .. +1 stronger). The stored analysis is what the forecaster uses,
so the signal can be recomputed later.

Signals: a currency's news pressure is a recency-weighted average of headline
scores, shrunk toward zero when there are few headlines. A pair's signal is
base pressure minus quote pressure. Only headlines the system had stored
strictly before the forecast origin, and published before it, count.
"""

from __future__ import annotations

import html
import json
import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from .data import CURRENCIES, http_get
from .store import stable_id
from .timeutil import UTC, iso, parse_iso

ANALYZER = "lexicon-v1"
TAU_HOURS = 12.0
LOOKBACK_HOURS = 48.0
SHRINK = 3.0
MAX_AGE_AT_FETCH = timedelta(days=3)


def _gnews(q: str, lang: str) -> str:
    if lang == "ja":
        return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=ja&gl=JP&ceid=JP:ja"
    return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"


SOURCES = [
    # id, name, url, lang, weight, default currency (official feeds)
    ("gn-ja-fx", "Google News (為替)", _gnews("ドル円 OR 円相場 OR 為替 when:1d", "ja"), "ja", 1.0, None),
    ("gn-ja-boj", "Google News (日銀・介入)", _gnews("日銀 OR 植田総裁 OR 為替介入 OR 財務官 when:1d", "ja"), "ja", 1.0, None),
    ("gn-ja-us", "Google News (米国)", _gnews("FRB OR FOMC OR 米雇用統計 OR 米CPI OR 米長期金利 when:1d", "ja"), "ja", 1.0, None),
    ("gn-ja-world", "Google News (欧州・英・豪)", _gnews("ユーロ OR ECB OR ポンド OR 英中銀 OR 豪ドル OR 豪中銀 when:1d", "ja"), "ja", 1.0, None),
    ("gn-en-fx", "Google News (FX)", _gnews("(dollar OR yen OR euro OR sterling OR \"Australian dollar\") forex when:1d", "en"), "en", 1.0, None),
    ("gn-en-cb", "Google News (central banks)", _gnews("(\"Federal Reserve\" OR \"Bank of Japan\" OR ECB OR \"Bank of England\" OR RBA) rates when:1d", "en"), "en", 1.0, None),
    ("gn-en-macro", "Google News (macro/risk)", _gnews("(tariffs OR geopolitical OR sanctions OR election OR \"Treasury yields\") markets when:1d", "en"), "en", 0.8, None),
    ("forexlive", "ForexLive", "https://www.forexlive.com/feed/news", "en", 1.0, None),
    ("fed", "Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml", "en", 1.2, "USD"),
    ("ecb", "European Central Bank", "https://www.ecb.europa.eu/rss/press.html", "en", 1.2, "EUR"),
    ("boj", "日本銀行", "https://www.boj.or.jp/rss/whatsnew.xml", "ja", 1.2, "JPY"),
    ("boe", "Bank of England", "https://www.bankofengland.co.uk/rss/news", "en", 1.2, "GBP"),
    ("rba", "Reserve Bank of Australia", "https://www.rba.gov.au/rss/rss-cb-media-releases.xml", "en", 1.2, "AUD"),
]
SOURCE_WEIGHT = {s[0]: s[4] for s in SOURCES}
CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


# ------------------------------------------------------------------ parsing

_NS = {"a": "http://www.w3.org/2005/Atom", "rss1": "http://purl.org/rss/1.0/",
       "dc": "http://purl.org/dc/elements/1.1/"}
_TAG = re.compile(r"<[^>]+>")


def _text(s: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub(" ", s or ""))).strip()


def _when(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        return parsedate_to_datetime(s).astimezone(UTC)
    except (TypeError, ValueError):
        pass
    try:
        return parse_iso(s)
    except ValueError:
        return None


def parse_feed(raw: bytes) -> list[dict]:
    """RSS 2.0, Atom or RDF (RSS 1.0) -> [{title, link, published, publisher}]."""
    root = ET.fromstring(raw)
    out = []
    for it in root.findall(".//item"):
        src = it.find("source")
        out.append({
            "title": _text(it.findtext("title")),
            "link": (it.findtext("link") or "").strip(),
            "published": _when(it.findtext("pubDate") or it.findtext("dc:date", namespaces=_NS)),
            "publisher": _text(src.text) if src is not None and src.text else "",
        })
    for it in root.findall(".//a:entry", _NS):
        link = it.find("a:link", _NS)
        out.append({
            "title": _text(it.findtext("a:title", namespaces=_NS)),
            "link": (link.get("href") if link is not None else "") or "",
            "published": _when(it.findtext("a:published", namespaces=_NS) or it.findtext("a:updated", namespaces=_NS)),
            "publisher": "",
        })
    for it in root.findall(".//rss1:item", _NS):
        out.append({
            "title": _text(it.findtext("rss1:title", namespaces=_NS)),
            "link": (it.findtext("rss1:link", namespaces=_NS) or "").strip(),
            "published": _when(it.findtext("dc:date", namespaces=_NS)),
            "publisher": "",
        })
    return [o for o in out if o["title"]]


def _clean_title(title: str, publisher: str) -> str:
    if publisher and title.endswith(" - " + publisher):
        return title[: -len(publisher) - 3].strip()
    return title


def _norm_key(title: str) -> str:
    return re.sub(r"[\W_]+", "", title.lower())


def normalise(entries: list[dict], source: tuple, now: datetime) -> list[dict]:
    sid, sname, _url, lang, _w, default_cur = source
    out = []
    for e in entries:
        pub = e["published"] or now
        pub = min(pub, now)
        if now - pub > MAX_AGE_AT_FETCH:
            continue
        title = _clean_title(e["title"], e["publisher"])
        item = {
            "id": stable_id(_norm_key(title)),
            "src": sid,
            "publisher": e["publisher"] or sname,
            "title": title[:300],
            "link": e["link"][:600],
            "published_at": iso(pub),
            "lang": lang,
        }
        item["an"] = analyze_lexicon(title, lang, default_cur)
        out.append(item)
    return out


# ---------------------------------------------------------------- analysis

EN_ENT = {
    "USD": [r"\bu\.?s\.? dollars?\b", r"\bgreenback\b", r"(?<!australian )(?<!canadian )(?<!zealand )(?<!hong kong )(?<!singapore )(?<!taiwan )\bdollar\b",
            r"\busd\b", r"\bdxy\b", r"\bfed\b", r"\bfederal reserve\b", r"\bfomc\b", r"\bpowell\b",
            r"\btreasur(?:y|ies)\b", r"\bnonfarm\b", r"\bpayrolls\b", r"\bu\.?s\.?\b", r"\bamerica\b", r"\btrump\b"],
    "JPY": [r"\byen\b", r"\bjpy\b", r"\bbank of japan\b", r"\bboj\b", r"\bueda\b", r"\bjapan(?:ese|'s)?\b", r"\bjgbs?\b",
            r"\btakaichi\b"],
    "EUR": [r"\beuros?\b(?!pe)", r"\beur\b", r"\becb\b", r"\blagarde\b", r"\beuro ?zone\b", r"\beuro area\b", r"\bbunds?\b",
            r"\bgerman(?:y|'s)?\b", r"\bfrance\b", r"\bfrench\b"],
    "GBP": [r"\bpound\b", r"\bsterling\b", r"\bgbp\b", r"\bcable\b", r"\bbank of england\b", r"\bboe\b", r"\bbailey\b",
            r"\bgilts?\b", r"\bu\.?k\.?\b", r"\bbritain\b", r"\bbritish\b"],
    "AUD": [r"\baussie\b", r"\baustralian dollar\b", r"\baud\b", r"\brba\b", r"\breserve bank of australia\b",
            r"\bbullock\b", r"\baustralia(?:n|'s)?\b"],
    "OTHER": [r"\brupee\b", r"\byuan\b", r"\brenminbi\b", r"\bwon\b", r"\bpeso\b", r"\blira\b", r"\bro?ubles?\b",
              r"\bloonie\b", r"\bcanadian dollar\b", r"\bkiwi\b", r"\bnew zealand dollar\b", r"\bfranc\b",
              r"\bringgit\b", r"\bbaht\b", r"\brupiah\b", r"\breal\b", r"\bbitcoin\b", r"\bgold\b", r"\boil\b",
              r"\bstocks?\b", r"\bshares\b", r"\bnikkei\b", r"\bs&p\b", r"\bnasdaq\b", r"\bbrent\b"],
}
EN_PAIR = re.compile(r"\b(usd|eur|gbp|aud)\s?/?\s?(jpy|usd)\b")
EN_UP = r"(?:rises|rose|rising|gains|gained|climbs|climbed|jumps|jumped|surges|surged|rall(?:y|ies|ied)|strengthens|strengthened|firms|firmed|rebounds|rebounded|advances|advanced|soars|soared|recovers|recovered|extends gains|higher|(?:hits|at|near) [\w\- ]{0,15}highs?)"
EN_DOWN = r"(?:falls|fell|falling|drops|dropped|slides|slid|slips|slipped|sinks|sank|tumbles|tumbled|plunges|plunged|weakens|weakened|declines|declined|retreats|retreated|slumps|slumped|dips|dipped|loses|lost|lower|extends losses|under pressure|(?:hits|at|near) [\w\- ]{0,15}lows?)"
EN_VERB = None  # compiled below
EN_OBJ_DOWN = r"(?:pressures?|pressured|weighs? on|weighed on|hurts?|drags?(?: down)?|knocks?|hammers?|batters?)\s+(?:the\s+)?"
EN_OBJ_UP = r"(?:lifts?|boosts?|supports?|buoys?|props? up|underpins?|bolsters?)\s+(?:the\s+)?"
EN_VERB = re.compile(r"\b(?:(?P<up>" + EN_UP + r")|(?P<down>" + EN_DOWN + r"))")
EN_PAIR_VERB = re.compile(r"[\s:,\-]*(?:(?:rate|pair|exchange rate|price)\s+)?(?:(?P<up>" + EN_UP + r")|(?P<down>" + EN_DOWN + r"))")
EN_HAWK = r"\b(?:hikes?|hiking|hiked|raises? rates|rate (?:rise|increase|hike)s?|tighten(?:s|ing)?|hawkish|higher for longer|inflation (?:rises|jumps|accelerates|surges|heats|hotter)|sticky inflation)\b"
EN_DOVE = r"\b(?:rate cuts?|cuts? rates|cutting|lowers? rates|easing|eases policy|dovish|stimulus|recession|slowdown|inflation (?:cools|eases|slows|falls)|disinflation)\b"
EN_BEAT = r"\b(?:beats?|tops) (?:expectations|estimates|forecasts)\b|\b(?:better|stronger)[- ]than[- ]expected\b|\babove (?:expectations|forecasts|estimates)\b"
EN_MISS = r"\bmiss(?:es|ed)? (?:expectations|estimates|forecasts)\b|\b(?:worse|weaker|softer)[- ]than[- ]expected\b|\bbelow (?:expectations|forecasts|estimates)\b"
EN_YIELD_UP = r"\b(?:treasury|u\.?s\.?|us|10-year|bond) yields? (?:rise|rises|rose|jump|jumps|jumped|climb|climbs|surge|surges|surged|soar|soars|higher|spike|spikes|hit)"
EN_YIELD_DOWN = r"\b(?:treasury|u\.?s\.?|us|10-year|bond) yields? (?:fall|falls|fell|drop|drops|dropped|slide|slides|lower|tumble|tumbles|ease|eases)"
EN_RISK_OFF = r"\b(?:war|attacks?|missiles?|airstrikes?|conflict|invasion|sanctions|crisis|turmoil|sell-?off|crash|panic|safe[- ]haven|geopolitic\w*|tensions|escalat\w*|trade war|shutdown|default)\b"
EN_RISK_ON = r"\b(?:stocks? rally|risk appetite|risk-on|ceasefire|truce|trade deal|deal reached)\b"
EN_INTERVENE = r"\b(?:interven\w+|rate checks?)\b"
EN_POLITICS = r"\b(?:election|snap poll|resign\w*|no-confidence|political (?:crisis|turmoil|uncertainty)|government collapse|impeach\w*)\b"

JA_ENT = {
    "USD": ["米ドル", "米国", "米連邦", "FRB", "FOMC", "パウエル", "米金利", "米長期金利", "米国債", "米雇用", "米CPI",
            "米消費者物価", "米経済", "トランプ", "米財務", "米GDP", "米小売", "米"],
    "JPY": ["円相場", "円安", "円高", "円買い", "円売り", "円急落", "円急伸", "円急騰", "円反発", "円反落", "円続落",
            "円続伸", "円上昇", "円下落", "円キャリー", "日銀", "日本銀行", "植田", "為替介入", "財務省", "財務官",
            "日本国債", "高市", "日本経済"],
    "EUR": ["ユーロ", "ECB", "欧州中銀", "欧州中央銀行", "ラガルド", "ユーロ圏", "ドイツ", "欧州"],
    "GBP": ["ポンド", "英国", "英中銀", "イングランド銀行", "BOE", "ベイリー", "英"],
    "AUD": ["豪ドル", "豪州", "オーストラリア", "豪中銀", "豪準備銀行", "RBA", "ブロック総裁"],
}
JA_MOVES = [  # longest first; (text, {currency: value})
    ("円安修正", {"JPY": 1}), ("円安是正", {"JPY": 1}), ("円安一服", {"JPY": 0.5}),
    ("円高修正", {"JPY": -1}), ("円高一服", {"JPY": -0.5}),
    ("ドル高一服", {"USD": -0.5}), ("ドル安一服", {"USD": 0.5}),
    ("豪ドル高", {"AUD": 1}), ("豪ドル安", {"AUD": -1}),
    ("ユーロ高", {"EUR": 1}), ("ユーロ安", {"EUR": -1}),
    ("ポンド高", {"GBP": 1}), ("ポンド安", {"GBP": -1}),
    ("ドル高", {"USD": 1}), ("ドル安", {"USD": -1}),
    ("円急落", {"JPY": -1}), ("円続落", {"JPY": -1}), ("円下落", {"JPY": -1}), ("円売り", {"JPY": -1}),
    ("円反落", {"JPY": -1}), ("円安", {"JPY": -1}),
    ("円急伸", {"JPY": 1}), ("円急騰", {"JPY": 1}), ("円反発", {"JPY": 1}), ("円上昇", {"JPY": 1}),
    ("円続伸", {"JPY": 1}), ("円買い", {"JPY": 1}), ("円高", {"JPY": 1}),
]
JA_PAIRS = [("豪ドル円", "AUD", "JPY"), ("ユーロ円", "EUR", "JPY"), ("ポンド円", "GBP", "JPY"),
            ("ユーロドル", "EUR", "USD"), ("ポンドドル", "GBP", "USD"), ("豪ドル米ドル", "AUD", "USD"),
            ("ドル円", "USD", "JPY"), ("ドル・円", "USD", "JPY")]
# "円は下落", "ユーロが上昇": currency, optional particle, then the move.
JA_CUR_MOVE = re.compile(r"(豪ドル|ユーロ|ポンド|ドル|円)(?:相場)?(?:は|が|も)\s?"
                         r"(?:(?P<down>急落|続落|下落|反落|軟調|売られ|弱含)|(?P<up>急伸|急騰|続伸|上昇|反発|堅調|買われ|強含))")
JA_CUR_CODE = {"豪ドル": "AUD", "ユーロ": "EUR", "ポンド": "GBP", "ドル": "USD", "円": "JPY"}
JA_PAIR_UP = re.compile(r"^(?:相場)?(?:が|は)?\s?(?:上昇|続伸|急伸|反発|高|上伸)")
JA_PAIR_DOWN = re.compile(r"^(?:相場)?(?:が|は)?\s?(?:下落|続落|急落|反落|安|下押し)")
JA_HAWK = r"利上げ|引き締め|タカ派|インフレ加速|物価上昇|物価高"
JA_DOVE = r"利下げ|金融緩和|緩和|ハト派|景気後退|景気減速|減速|物価下落"
JA_BEAT = r"予想(?:を)?上回|上振れ|好調|改善"
JA_MISS = r"予想(?:を)?下回|下振れ|悪化|低迷"
JA_YIELD_UP = r"米(?:国)?の?(?:長期)?金利(?:が|は)?[^、。]{0,6}(?:上昇|高水準|最高)|米国債利回り(?:が|は)?[^、。]{0,4}上昇"
JA_YIELD_DOWN = r"米(?:国)?の?(?:長期)?金利(?:が|は)?[^、。]{0,6}(?:低下|下落)|米国債利回り(?:が|は)?[^、。]{0,4}低下"
JA_RISK_OFF = r"地政学|紛争|戦争|攻撃|ミサイル|暴落|リスクオフ|有事|緊張|制裁(?!金)|関税"
JA_RISK_ON = r"株高|最高値|リスクオン|停戦|合意"
JA_INTERVENE = r"為替介入|介入|レートチェック"
JA_POLITICS = r"解散|総選挙|辞任|政局|不信任"

RISK_OFF_EFFECT = {"JPY": 0.4, "USD": 0.15, "AUD": -0.4}
RISK_ON_EFFECT = {"JPY": -0.3, "AUD": 0.3}
_OTHER_DOLLARS = ("豪", "NZ", "加", "カナダ", "香港", "シンガポール", "台湾")


def _entities_en(t: str) -> list[tuple[int, int, str]]:
    ents = []
    for cur, pats in EN_ENT.items():
        for p in pats:
            for m in re.finditer(p, t):
                ents.append((m.start(), m.end(), cur))
    # Longer matches win where two overlap ("australian dollar" over "dollar").
    ents.sort(key=lambda e: (e[0], -(e[1] - e[0])))
    out, last_end = [], -1
    for e in ents:
        if e[0] >= last_end:
            out.append(e)
            last_end = e[1]
    return out


def _entities_ja(t: str) -> list[tuple[int, int, str]]:
    ents = []
    for cur, words in JA_ENT.items():
        for w in words:
            start = 0
            while (i := t.find(w, start)) >= 0:
                ents.append((i, i + len(w), cur))
                start = i + 1
    i = 0
    while (i := t.find("ドル", i)) >= 0:
        if not any(t[max(0, i - len(p)):i] == p for p in _OTHER_DOLLARS) and not t[max(0, i - 1):i] == "米":
            ents.append((i, i + 2, "USD"))
        i += 2
    ents.sort(key=lambda e: (e[0], -(e[1] - e[0])))
    out, last_end = [], -1
    for e in ents:
        if e[0] >= last_end:
            out.append(e)
            last_end = e[1]
    return out


def _nearest(ents, pos: int, max_dist: int = 10_000, tracked_only: bool = True):
    best, dist = None, max_dist + 1
    for s, e, cur in ents:
        if tracked_only and cur not in CURRENCIES:
            continue
        d = 0 if s <= pos < e else min(abs(pos - s), abs(pos - e))
        if d < dist:
            best, dist = cur, d
    return best


def analyze_lexicon(title: str, lang: str, default_cur: str | None = None) -> dict:
    """Keyword analysis of one headline -> {"by", "cur": {currency: score}, "men": [...], "top": [...]}."""
    effects: dict[str, float] = {}
    topics: set[str] = set()

    def add(cur, v, topic):
        if cur in CURRENCIES and v:
            effects[cur] = effects.get(cur, 0.0) + v
            topics.add(topic)

    if lang == "ja":
        t = title
        ents = _entities_ja(t)
        masked = t
        for word, eff in JA_MOVES:
            while (i := masked.find(word)) >= 0:
                for cur, v in eff.items():
                    add(cur, v, "fx_move")
                masked = masked[:i] + "＿" * len(word) + masked[i + len(word):]
        # Pair names ("ドル円") are scored as pairs below, not as their component currencies.
        single = masked
        for word, _, _ in JA_PAIRS:
            single = single.replace(word, "＿" * len(word))
        for m in JA_CUR_MOVE.finditer(single):
            cur = JA_CUR_CODE[m.group(1)]
            if cur == "USD" and m.start() > 0 and single[m.start() - 1] in "豪米加":
                cur = "USD" if single[m.start() - 1] == "米" else None
            add(cur, 1 if m.group("up") else -1, "fx_move")
        for word, base, quote in JA_PAIRS:
            start = 0
            while (i := t.find(word, start)) >= 0:
                rest = t[i + len(word): i + len(word) + 8]
                d = 1 if JA_PAIR_UP.match(rest) else -1 if JA_PAIR_DOWN.match(rest) else 0
                if d:
                    add(base, d, "fx_move")
                    add(quote, -d, "fx_move")
                start = i + len(word)
        cues = [(JA_HAWK, 0.8, "policy"), (JA_DOVE, -0.8, "policy"), (JA_BEAT, 0.6, "data"),
                (JA_MISS, -0.6, "data"), (JA_POLITICS, -0.3, "politics")]
        yield_up, yield_down = JA_YIELD_UP, JA_YIELD_DOWN
        risk_off, risk_on, intervene = JA_RISK_OFF, JA_RISK_ON, JA_INTERVENE
    else:
        t = title.lower()
        pairs = list(EN_PAIR.finditer(t))
        # Tokens inside "USD/JPY" belong to the pair, not to one currency.
        ents = [e for e in _entities_en(t) if not any(p.start() <= e[0] < p.end() for p in pairs)]
        for s, e, cur in ents:
            d = 0
            m = EN_VERB.search(t, e, e + 40)
            if m and m.start() - e <= 25 and not any(e <= s2 < m.start() for s2, _, _ in ents):
                d = 1 if m.group("up") else -1
            before = t[max(0, s - 14):s]
            if re.search(r"\b(?:stronger|firmer)\s+(?:the\s+)?$", before):
                d = 1
            elif re.search(r"\b(?:weaker|softer)\s+(?:the\s+)?$", before):
                d = -1
            if re.search(EN_OBJ_DOWN + r"$", t[max(0, s - 24):s]):
                d = -1
            elif re.search(EN_OBJ_UP + r"$", t[max(0, s - 24):s]):
                d = 1
            if not d:
                continue
            if cur == "OTHER":
                # "rupee falls against the dollar": the dollar moved the other way.
                if re.search(r"against (?:the )?(?:u\.?s\.? |us )?dollar", t[e:e + 60]):
                    add("USD", -0.5 * d, "fx_move")
                continue
            add(cur, d, "fx_move")
        for p in pairs:
            m = EN_PAIR_VERB.match(t, p.end(), p.end() + 40)
            if not m:
                continue
            d = 1 if m.group("up") else -1
            base, quote = p.group(1).upper(), p.group(2).upper()
            add(base, d, "fx_move")
            add(quote, -d, "fx_move")
        cues = [(EN_HAWK, 0.8, "policy"), (EN_DOVE, -0.8, "policy"), (EN_BEAT, 0.6, "data"),
                (EN_MISS, -0.6, "data"), (EN_POLITICS, -0.4, "politics")]
        yield_up, yield_down = EN_YIELD_UP, EN_YIELD_DOWN
        risk_off, risk_on, intervene = EN_RISK_OFF, EN_RISK_ON, EN_INTERVENE

    for pattern, value, topic in cues:
        for m in re.finditer(pattern, t):
            cur = _nearest(ents, m.start()) or default_cur
            add(cur, value, topic)
    if re.search(yield_up, t):
        add("USD", 0.6, "yields")
    if re.search(yield_down, t):
        add("USD", -0.6, "yields")
    if re.search(risk_off, t):
        for cur, v in RISK_OFF_EFFECT.items():
            add(cur, v, "risk")
    if re.search(risk_on, t):
        for cur, v in RISK_ON_EFFECT.items():
            add(cur, v, "risk")
    mentioned = sorted({c for _, _, c in ents if c in CURRENCIES} | ({default_cur} if default_cur else set()))
    # An intervention mention supports the yen unless the headline already says how the yen moved.
    if re.search(intervene, t) and "JPY" in mentioned and "fx_move" not in topics:
        add("JPY", 0.8, "intervention")
    scores = {c: round(math.tanh(v / 1.5), 3) for c, v in sorted(effects.items()) if abs(v) > 1e-9}
    return {"by": ANALYZER, "cur": scores, "men": mentioned, "top": sorted(topics)}


def is_relevant(item: dict) -> bool:
    an = item["an"]
    return bool(an["cur"]) or (item["src"] in ("fed", "ecb", "boj", "boe", "rba"))


# ---------------------------------------------------------------- fetching

def _fetch_source(src: tuple, now: datetime) -> tuple[str, list[dict], str | None]:
    try:
        raw = http_get(src[2], timeout=20, retries=2, accept="application/rss+xml, application/xml, text/xml")
        return src[0], normalise(parse_feed(raw), src, now), None
    except Exception as exc:
        return src[0], [], f"{src[0]}: {exc}"


def collect_news(now: datetime, sources=SOURCES, workers: int = 8) -> tuple[list[dict], dict]:
    """Fetch every source in parallel; returns relevant de-duplicated items and a per-source report."""
    report: dict[str, dict] = {}
    seen: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda s: _fetch_source(s, now), sources))
    for sid, items, err in results:
        kept = [it for it in items if is_relevant(it)]
        report[sid] = {"fetched": len(items), "relevant": len(kept), "error": err}
        for it in kept:
            seen.setdefault(it["id"], it)
    return sorted(seen.values(), key=lambda it: (it["published_at"], it["id"])), report


def parse_calendar(payload: list[dict]) -> list[dict]:
    out = []
    for ev in payload:
        cur = ev.get("country")
        impact = ev.get("impact")
        if cur not in CURRENCIES or impact not in ("High", "Medium"):
            continue
        try:
            when = datetime.fromisoformat(ev["date"]).astimezone(UTC)
        except (KeyError, ValueError):
            continue
        out.append({
            "id": stable_id(cur, ev.get("title", ""), iso(when)),
            "cur": cur,
            "title": ev.get("title", ""),
            "time": iso(when),
            "impact": impact,
            "forecast": ev.get("forecast", ""),
            "previous": ev.get("previous", ""),
        })
    return out


def collect_calendar() -> tuple[list[dict], str | None]:
    try:
        return parse_calendar(json.loads(http_get(CALENDAR_URL, accept="application/json"))), None
    except Exception as exc:
        return [], f"calendar: {exc}"


# ----------------------------------------------------------------- signals

def eligible(items: list[dict], cutoff: datetime) -> list[dict]:
    """Headlines stored strictly before ``cutoff``, published at or before it, within the lookback."""
    c = iso(cutoff)
    lo = iso(cutoff - timedelta(hours=LOOKBACK_HOURS))
    return [it for it in items if it["fetched_at"] < c and lo <= it["published_at"] <= c]


def pressures(items: list[dict], cutoff: datetime) -> dict[str, dict]:
    num = {c: 0.0 for c in CURRENCIES}
    den = {c: 0.0 for c in CURRENCIES}
    cnt = {c: 0 for c in CURRENCIES}
    for it in eligible(items, cutoff):
        age_h = (cutoff - parse_iso(it["published_at"])).total_seconds() / 3600.0
        d = SOURCE_WEIGHT.get(it["src"], 0.8) * math.exp(-age_h / TAU_HOURS)
        for cur, score in it["an"]["cur"].items():
            num[cur] += d * score
            den[cur] += d
            cnt[cur] += 1
    return {c: {"p": round(num[c] / (SHRINK + den[c]), 6), "w": round(den[c], 4), "n": cnt[c]} for c in CURRENCIES}


def pair_signal(press: dict[str, dict], base: str, quote: str) -> float:
    return round(max(-1.0, min(1.0, press[base]["p"] - press[quote]["p"])), 6)


def events_between(events: list[dict], currencies, start: datetime, end: datetime, known_before: datetime) -> list[dict]:
    """Calendar events for ``currencies`` scheduled in (start, end] and known before ``known_before``."""
    s, e, k = iso(start), iso(end), iso(known_before)
    out = []
    for ev in events:
        if ev["cur"] in currencies and s < ev["time"] <= e and ev["fetched_at"] < k:
            out.append({"time": parse_iso(ev["time"]), "impact": ev["impact"], "cur": ev["cur"], "title": ev["title"]})
    return out
