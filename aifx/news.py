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
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

from .data import CURRENCIES, http_get
from .store import stable_id
from .timeutil import UTC, iso, parse_iso

ANALYZER = "lexicon-v4"
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

# lexicon-v4 (research/news_v4.md): lexicon-v3 with the systematic errors found on hand-labelled
# headlines fixed; lexicon-v3 is kept in news_v3.py.
# ------------------------------------------------------------------ English

EN_ENT = {
    "USD": [r"\bu\.?s\.? dollars?\b", r"\bgreenback'?s?\b",
            r"(?<!australian )(?<!canadian )(?<!zealand )(?<!hong kong )(?<!singapore )(?<!taiwan )(?<!aussie )"
            r"(?<!zimbabwe )(?<!jamaican )(?<!namibian )(?<!fiji )(?<!liberian )(?<!brunei )\bdollar\b",
            r"\busd\b", r"\bdxy\b", r"\bdollar index\b", r"(?<![\w\-])fed\b(?! govt| government| up\b| ex\b)",
            r"\bfederal reserve\b", r"\bfomc\b", r"\b(?:jerome|fed chair(?:man)?|chair) powell\b",
            r"\btreasur(?:y|ies)\b", r"\bnonfarm\b", r"\bpayrolls\b", r"\bu\.s\.?(?=\W|$)", r"\bus(?=\W|$)",
            r"\bwarsh\b", r"\bbessent\b", r"\bamerican economy\b"],
    "JPY": [r"\byen\b", r"\bjpy\b", r"\bbank of japan\b", r"\bboj\b", r"\bueda\b", r"\bjapan(?:ese|'s)?\b", r"\bjgbs?\b",
            r"\btakaichi\b", r"\bkatayama\b", r"\bmimura\b"],
    "EUR": [r"\beuros?\b(?!pe)", r"\beur\b", r"\becb\b", r"\blagarde\b", r"\beuro ?zone\b", r"\beuro area\b", r"\bbunds?\b",
            r"\bgerman(?:y|'s)?\b", r"\bfrance\b", r"\bfrench\b"],
    "GBP": [r"\bpound\b", r"\bsterling\b", r"\bgbp\b", r"\bcable\b", r"\bbank of england\b", r"\bboe\b", r"\bbailey\b",
            r"\bgilts?\b", r"\bu\.?k\.?\b", r"\bbritain\b", r"\bbritish\b", r"\breeves\b"],
    "AUD": [r"\baussie\b", r"\baustralian dollar\b", r"\baud\b", r"\brba\b", r"\breserve bank of australia\b",
            r"\bbullock\b", r"\baustralia(?:n|'s)?\b", r"\bchalmers\b"],
    # other currencies, countries and central banks: a move or a policy cue next to them is theirs
    "OTHER": [r"\brupees?\b", r"\byuan\b", r"\brenminbi\b", r"\bwon\b", r"\bpesos?\b", r"\blira\b", r"\bro?ubles?\b",
              r"\bloonie\b", r"\bcanadian dollar\b", r"\bkiwi\b", r"\bnew zealand dollar\b", r"\bfrancs?\b",
              r"\bringgit\b", r"\bbaht\b", r"\brupiah\b", r"\breal\b", r"\bnaira\b", r"\brand\b", r"\bcedi\b",
              r"\bshillings?\b", r"\bdong\b", r"\btaka\b", r"\bdirham\b", r"\briyal\b", r"\bzloty\b", r"\bforint\b",
              r"\bkoruna\b", r"\bshekel\b", r"\bhryvnia\b", r"\bkwacha\b", r"\btenge\b", r"\btaiwan dollar\b",
              r"\bhong kong dollar\b", r"\bsingapore dollar\b",
              r"\bbitcoin\b", r"\bgold\b", r"\bsilver\b", r"\boil\b", r"\bcrude\b", r"\bbrent\b",
              r"\bstocks?\b", r"\bshares\b", r"\bnikkei\b", r"\bs&p\b", r"\bnasdaq\b", r"\bsensex\b", r"\bnifty\b"],
    "ELSEWHERE": [r"\bindia(?:n|'s)?\b", r"\bchina(?:'s)?\b", r"\bchinese\b", r"\bkorea(?:n|'s)?\b", r"\bcanada(?:'s)?\b",
                  r"\bcanadian\b", r"\bbrazil\w*\b", r"\bmexic\w*\b", r"\bturk\w*\b", r"\brussia\w*\b",
                  r"\bindonesia\w*\b", r"\bmalaysia\w*\b", r"\bphilippine\w*\b", r"\bnigeria\w*\b", r"\bpakistan\w*\b",
                  r"\bsouth africa\w*\b", r"\bnew zealand\b", r"\bswiss\b", r"\bswitzerland\b", r"\bsweden\b",
                  r"\bnorw\w*\b", r"\bthai\w*\b", r"\bvietnam\w*\b", r"\bpoland\b", r"\bhungar\w*\b", r"\begypt\w*\b",
                  r"\brbi\b", r"\bpboc\b", r"\bbank of canada\b", r"\bboc\b", r"\bsnb\b", r"\brbnz\b",
                  r"\bbank of korea\b", r"\bbok\b", r"\bbanxico\b", r"\bcbrt\b", r"\bnorges bank\b", r"\briksbank\b",
                  r"\bbsp\b", r"\bbnm\b"],
}
TRACKED_OR_ELSEWHERE = set(CURRENCIES) | {"ELSEWHERE"}
# ambiguous names need a context: the currency / monetary topic of the headline
EN_FX_CTX = re.compile(r"\b(?:forex|fx|currenc\w*|exchange rates?|dollar|greenback|usd|yen|jpy|euro|eur|gbp|aud|sterling|"
                       r"pound|boe|bank of england|gilts?|cable|bailey|reeves|aussie dollar|australian dollar|interven\w*)\b")
EN_MON_CTX = re.compile(r"\b(?:rates?|hikes?|hiking|hiked|cuts?|cutting|inflation|monetary|policy|governor|central bank|mpc|"
                        r"fomc|federal reserve|bank of england|reserve bank|rba|boe|ecb|boj|pound|sterling|gilts?|yields?|"
                        r"economy|economic|recession|mortgages?|pill|lombardelli|hawkish|dovish)\b")
_NEEDS_FX = {"pound", "sterling", "cable"}
_NEEDS_MON = {"boe", "bailey", "bullock"}
_NUMBER_BEFORE = re.compile(r"\d[\d,.]*\s?-?\s?$")

EN_G10 = "usd|eur|gbp|aud|jpy|cad|chf|nzd|sek|nok"
EN_EM = "cny|cnh|inr|krw|try|mxn|zar|brl|rub|hkd|sgd|twd|thb|idr|php|myr|pln|huf|czk|ils|ngn|pkr|vnd|egp|kes"
EN_PAIR = re.compile(r"\b(" + EN_G10 + "|" + EN_EM + r")\s?[/\-]?\s?(" + EN_G10 + "|" + EN_EM + r")\b")
EN_WORD_CUR = {"pound": "GBP", "sterling": "GBP", "euro": "EUR", "dollar": "USD", "us dollar": "USD", "yen": "JPY",
               "aussie": "AUD", "australian dollar": "AUD"}
EN_WORD_PAIR = re.compile(r"\b(pound|sterling|euro|us dollar|dollar|yen|australian dollar|aussie)"
                          r"(?:\s?-?\s?to\s?-?\s?|\s?/\s?|\s(?:vs\.?|versus)\s|\s)"
                          r"(pound|sterling|euro|us dollar|dollar|yen|australian dollar|aussie)\b")

EN_UP = (r"(?:rises|rose|rising|gains|gained|climbs|climbed|jumps|jumped|surges|surged|rall(?:y|ies|ied)|strengthens|"
         r"strengthened|firms|firmed|rebounds|rebounded|advances|advanced|soars|soared|recovers|recovered|extends gains|higher|"
         r"spikes|spiked|leaps|leapt|pops|bounces|bounced|(?:hits|at|near|reaches|reached|touches|touched|to) [\w\- ]{0,15}(?:highs?|peaks?))")
EN_DOWN = (r"(?:falls|fell|falling|drops|dropped|slides|slid|slips|slipped|sinks|sank|tumbles|tumbled|plunges|plunged|"
           r"weakens|weakened|declines|declined|retreats|retreated|slumps|slumped|dips|dipped|loses|lost|lower|extends losses|"
           r"under pressure|struggles|struggling|tanks|tanked|skids|plummets|plummeted|stumbles|stumbled|sags|"
           r"(?:hits|at|near|reaches|reached|touches|touched|to) [\w\- ]{0,15}(?:lows?|troughs?))")
EN_OBJ_DOWN = r"(?:pressures?|pressured|weighs? on|weighed on|hurts?|drags?(?: down| on)?|knocks?|hammers?|batters?|dents?|saps?)\s+(?:the\s+)?"
EN_OBJ_UP = r"(?:lifts?|boosts?|supports?|buoys?|props? up|underpins?|bolsters?)\s+(?:the\s+)?"
EN_UP_MORE = r"edges? (?:up|higher)|inch(?:es|ed)? (?:up|higher)|ticks? (?:up|higher)|steadies|recoups|holds gains"
EN_DOWN_MORE = (r"eases|eased|softens|softened|edges? (?:down|lower)|inch(?:es|ed)? (?:down|lower)|ticks? (?:down|lower)|"
                r"pares gains|gives up gains|(?:hovers|holds|trades|stays) (?:near|around|at) [\w\- ]{0,15}lows?")
EN_VERB = re.compile(r"\b(?:(?P<up>" + EN_UP_MORE + "|" + EN_UP + r")|(?P<down>" + EN_DOWN_MORE + "|" + EN_DOWN + r"))")
EN_VIEW_UP = r"bullish|bulls?'?s?|upside|breakout|bid|eyes? (?:further )?gains|set to (?:rise|gain|climb|rally)"
EN_VIEW_DOWN = r"bearish|bears?'?s?|downside|breakdown|offered|eyes? (?:further )?losses|set to (?:fall|drop|slide|decline)"
EN_PAIR_VERB = re.compile(r"[\s:,\-]*(?:(?:rate|pair|exchange rate|price|(?:price |weekly |daily )?(?:forecast|outlook|analysis|"
                          r"prediction)s?)\s*[:\-–]?\s*)?(?:(?:today|now|still|further|again)\s+)?"
                          r"(?:(?P<up>" + EN_UP_MORE + "|" + EN_UP + "|" + EN_VIEW_UP + r")|(?P<down>" + EN_DOWN_MORE + "|"
                          + EN_DOWN + "|" + EN_VIEW_DOWN + r"))\b")
EN_VIEW = re.compile(r"\b(?:(?P<up>" + EN_VIEW_UP + r")|(?P<down>" + EN_VIEW_DOWN + r"))\b")
# nouns after a currency ("USD selloff accelerates", "yen keeps showing persistent weakness", "dollar recovery")
EN_NOUN_UP = r"strength|rally|gains|recovery|rebound|resilience|upswing"
EN_NOUN_DOWN = r"weakness|slump|slide|sell-?off|selloff|losses|decline|dip|drop|on the ropes|on the back foot"
EN_NOUN_AFTER = re.compile(r"[\s\-]+(?:[\w\-]+\s+){0,3}?(?:(?P<up>" + EN_NOUN_UP + r")|(?P<down>" + EN_NOUN_DOWN + r"))\b")
# "recovery in EUR/USD", "positive on the USD", "puts the euro on the ropes"
EN_NOUN_BEFORE = re.compile(r"\b(?:(?P<up>recovery|rebound|rally|gains|upside|bullish|positive|optimistic)|(?P<down>decline|drop|slide|"
                            r"losses|downside|bearish|negative|pessimistic))\s+(?:in|on|for)\s+(?:the\s+)?(?:u\.?s\.?\s+)?$")
EN_PAIR_LABEL = re.compile(r"[\s:,\-]*(?:rate|exchange rate|(?:price |weekly |daily |short-term |technical )?(?:forecast|outlook|"
                           r"analysis|prediction)s?)\b")
# a word between a currency and a verb that is the verb's real subject ("yen intervention weakens dollar")
EN_OTHER_SUBJECT = re.compile(r"\b(?:intervention\w*|bets?|data|fears?|hopes?|yields?|comments?|remarks?|policy|decision|carry|"
                              r"traders?|investors?|shorts?|longs?|positions?|demand|supply|deposits?|debt|bonds?|funds?|loans?|"
                              r"reserves?|assets?|prices?|inflows?|outflows?)\b")

# policy: "hike" alone is a rate hike only with a central bank or "rate" in the headline and not a price,
# tax or spending hike
EN_HAWK = (r"\b(?:rate[- ]hikes?|hikes? (?:interest |policy |its |the |key |benchmark )*rates?|hiking (?:interest )?rates?|"
           r"hiked (?:interest )?rates?|(?:raises?|raised|raising) (?:interest |policy |its |the |key |benchmark |cash )*rates?|"
           r"(?<!mortgage )(?<!exchange )(?<!unemployment )(?<!jobless )(?<!tax )(?<!growth )(?<!savings )(?<!deposit )"
           r"(?<!lending )(?<!loan )(?<!fixed )(?<!variable )(?<!participation )(?<!birth )(?<!crime )(?<!vacancy )"
           r"(?<!default )(?<!inflation )rate (?:rise|increase)s?|hawkish|hawks|higher for longer|"
           r"tighten(?:s|ing|ed)? (?:policy|monetary)|monetary tightening|policy tightening|"
           r"inflation (?:rises|rose|jumps|jumped|accelerates|accelerated|surges|surged|heats up|hotter|quickens|picks up|"
           r"climbs|climbed)|(?:sticky|hot|hotter|stubborn|persistent|accelerating) inflation)\b")
EN_HIKE_WORD = re.compile(r"\bhik(?:e|es|ed|ing)\b")
EN_HIKE_NOT = re.compile(r"\b(?:price|prices|tax|taxes|tariff|tariffs|budget|spending|defen[cs]e|wage|wages|fare|fares|toll|fees?|"
                         r"premiums?|rent|rents|salary|salaries|pay|energy|petrol|fuel|gas|electricity|vat|duty|duties|levy|"
                         r"insurance|subscription|ticket)\b")
EN_CB = re.compile(r"\b(?:fed|federal reserve|fomc|boj|bank of japan|ecb|boe|bank of england|rba|reserve bank|central banks?|"
                   r"warsh|ueda|lagarde|bailey|bullock|mpc|policymakers|rates?|dissenters)\b")
EN_DOVE = (r"\b(?:rate[- ]cuts?|cuts? (?:interest |policy |its |the |key |benchmark |cash )*rates?|cutting (?:interest |policy )?rates?|"
           r"lowers? (?:interest )?rates?|(?:monetary|policy|quantitative|further|more) easing|easing (?:cycle|bias)|"
           r"eases policy|dovish|recession|slowdown|"
           r"inflation (?:cools|cooled|eases|eased|slows|slowed|falls|fell|drops|dropped|declines|declined|softens|"
           r"softened|moderates|moderated|retreats|retreated)|(?:cooling|easing|slowing|falling|softer|cooler|weaker|"
           r"subdued) inflation|drop in (?:[\w\-]+ ){0,2}inflation|disinflation)\b")
EN_CB_CUT = r"\b(?:fed|boj|ecb|boe|rba|central bank)\s+cuts\b|\binflation (?:drop|decline|slowdown|miss)\b"
# "Will the Fed raise rates?": a policy cue in a question says nothing
EN_QUESTION = re.compile(r"(?:^|[.:!?|][\'’\"”]?\s*|\s[-–—]\s)(?:will|would|could|can|should|is|are|does|do|did|has|have)\b[^?.!|]*\?")
EN_MACRO = (r"(?:jobs?|payrolls?|nfp|employment|hiring|job growth|job gains|unemployment|gdp|growth|cpi|inflation|ppi|pce|"
            r"retail sales|consumer spending|pmi|ism|wages?|wage growth|consumer confidence|sentiment|output|exports|"
            r"economy|economic data|data|manufacturing|services|factory)")
EN_BEAT = (r"\b(?:beats?|tops) (?:expectations|estimates|forecasts)\b|\b(?:better|stronger|hotter)[- ]than[- ]expected\b|"
           r"\babove (?:expectations|forecasts|estimates)\b")
EN_MISS = (r"\bmiss(?:es|ed)? (?:expectations|estimates|forecasts)\b|\b(?:worse|weaker|softer|cooler)[- ]than[- ]expected\b|"
           r"\bbelow (?:expectations|forecasts|estimates)\b")
EN_DATA_UP = re.compile(r"\b" + EN_MACRO + r"\s+(?:[\w\-]+\s+){0,2}?(?:beats?|tops|surges?|surged|jumps?|jumped|accelerates?|"
                        r"accelerated|rebounds?|rebounded|booms?|expands?|expanded|strengthens?|smashes)\b|"
                        r"\b(?:strong|stronger|robust|solid|hot|blowout)\s+(?:us\s+|u\.s\.\s+)?" + EN_MACRO + r"\b")
EN_DATA_DOWN = re.compile(r"\b" + EN_MACRO + r"\s+(?:[\w\-]+\s+){0,2}?(?:miss(?:es|ed)?|slumps?|slumped|contracts?|contracted|"
                          r"shrinks?|shrank|slows?|slowed|cools?|cooled|weakens?|weakened|stalls?|stalled|disappoints?|"
                          r"disappointed)\b|\b(?:weak|weaker|soft|softer|dismal|poor|huge|big|negative)\s+(?:us\s+|u\.s\.\s+)?"
                          + EN_MACRO + r"(?:\s+(?:miss|report|data))?\b|\b(?:unemployment|jobless) (?:rate )?(?:rises|rose|"
                          r"jumps|jumped|climbs|climbed|hits)\b|\bjob(?:s)? miss\b|\bdowngrades? (?:[\w'\-]+ ){0,3}"
                          r"(?:growth|gdp|economic) (?:forecast|outlook)")
EN_DATA_UP_MORE = re.compile(r"\bupgrades? (?:[\w'\-]+ ){0,3}(?:growth|gdp|economic) (?:forecast|outlook)")
EN_US_YIELDS = r"(?:treasury|treasuries|u\.?s\.?|us|10-year|30-year|2-year|two-year|ten-year|bond)"
EN_YIELD_UP = (r"\b" + EN_US_YIELDS + r" yields? (?:(?:are|were|is|keep|kept) )?(?:[\w\-]+ing )?"
               r"(?:rise|rises|rose|jump|jumps|jumped|climb|climbs|climbed|surge|surges|surged|soar|soars|soared|higher|"
               r"spike|spikes|spiked|hit|hits|top|tops|edge higher|edges higher|edge up|edges up|inch higher|inches higher|"
               r"tick up|ticks up|push higher|pushes higher|ripping higher|rip higher|advance|advances|to (?:[\w\-]+ ){0,3}highs?\b|"
               r"to (?:[\w\-]+ ){0,3}highest)"
               r"|\b(?:higher|rising|surging|soaring|spiking|climbing) (?:(?:u\.?s\.?|us|treasury|bond|10-year|30-year|"
               r"long-term|longer-dated) ){0,2}yields\b"
               r"|\btreasur(?:y|ies) (?:sell-?off|selloff|slump|rout)\b|\bu\.?s\.? bond (?:sell-?off|selloff|rout)\b")
EN_YIELD_DOWN = (r"\b" + EN_US_YIELDS + r" yields? (?:(?:are|were|is|keep|kept) )?(?:[\w\-]+ing )?"
                 r"(?:fall|falls|fell|drop|drops|dropped|slide|slides|slid|lower|tumble|tumbles|tumbled|ease|eases|eased|"
                 r"edge lower|edges lower|edge down|edges down|slip|slips|slipped|retreat|retreats|retreated|decline|declines|"
                 r"declined|sink|sinks|to (?:[\w\-]+ ){0,3}lows?\b|to (?:[\w\-]+ ){0,3}lowest)"
                 r"|\b(?:lower|falling|sliding|tumbling|easing|retreating) (?:(?:u\.?s\.?|us|treasury|bond|10-year|30-year|"
                 r"long-term|longer-dated) ){0,2}yields\b"
                 r"|\btreasur(?:y|ies) rally\b")
EN_YIELD_US_CTX = re.compile(r"\b(?:treasur(?:y|ies)|u\.s\.|us|america\w*|fed|federal reserve|wall street|dollar)\b")
EN_YIELD_NOT_US = re.compile(r"\b(?:gilts?|jgbs?|bunds?|japan\w*|u\.?k\.?|brit\w*|german\w*|euro\w*|ital\w*|french|france|"
                             r"australia\w*|aussie|china|chinese|india\w*|global|world)\b")

# risk: a broad stock-market move with a risk reason, or explicit safe-haven words
EN_BROAD = re.compile(r"\b(?:(?:global|world|asian?|asia-pacific|european?|us|u\.s\.|american|japanese|japan|tokyo|australian|"
                      r"aussie|uk|british|london|wall street|emerging[- ]market|regional)\s+(?:stocks?|shares|equities|markets?|"
                      r"bourses?|indices|indexes)|stocks|equities|stock markets?|equity markets?|global markets|world markets|"
                      r"financial markets|markets|wall street|dow(?: jones)?|s&p(?: 500)?|nasdaq|nikkei|topix|ftse(?: 100)?|dax|"
                      r"stoxx(?: 600)?|asx(?: 200)?|hang seng)\b")
EN_LOCAL_MARKET = re.compile(r"\b(?:sensex|nifty|kospi|seoul|psx|kse|jse|karachi|dhaka|colombo|manila|jakarta|nepse|taiex|"
                             r"ghana|nigeria\w*|kenya\w*|pakistan\w*|india\w*|bangladesh\w*|sri lanka\w*|vietnam\w*|philippine\w*|"
                             r"indonesia\w*|malaysia\w*|thai\w*|egypt\w*|saudi|turk\w*|korea\w*|taiwan\w*|chinese|china|shanghai|"
                             r"shenzhen|canadian|tsx|mexic\w*|brazil\w*)\b")
EN_MAJOR_MARKET = re.compile(r"\b(?:global|world|wall street|dow|s&p|nasdaq|nikkei|topix|ftse|dax|stoxx|europe\w*|asia\w*|"
                             r"u\.?s\.? stocks|us stocks|asx|hang seng)\b")
EN_MKT_DOWN = (r"sags?|sagged|falls?|fell|falling|drops?|dropped|dropping|slides?|slid|sliding|slips?|slipped|sinks?|sank|tumbles?|tumbled|"
               r"plunges?|plunged|plunging|slumps?|slumped|retreats?|retreated|declines?|declined|dips?|dipped|lower|losses|lose|"
               r"loses|lost|skids?|skid|dives?|dived|tanks?|tanked|crash\w*|sell-?off|selloff|rout|in (?:the )?red|sink into|"
               r"under pressure|hit hard|reel\w*|wobble\w*|shed|sheds|slammed|hammered|pummel\w*|battered|bleed\w*|sour\w*")
EN_MKT_UP = (r"rises?|rose|rising|gains?|gained|climbs?|climbed|jumps?|jumped|surges?|surged|soars?|soared|rall(?:y|ies|ied)|"
             r"rebounds?|rebounded|advances?|advanced|higher|record highs?|all-time highs?|best (?:day|week|month)|in the green|"
             r"recovers?|recovered|bounce\w*|cheer\w*|up\b")
EN_MKT_MOVE = re.compile(r"\b(?:(?P<down>" + EN_MKT_DOWN + r")|(?P<up>" + EN_MKT_UP + r"))\b")
EN_RISK_REASON = (r"\b(?:war|wars|attacks?|missiles?|airstrikes?|strikes on|conflict|invasion|sanctions|crisis|turmoil|panic|"
                  r"geopolitic\w*|tensions?|escalat\w*|tariffs?|trade war|shutdown|fears?|worr(?:y|ies|ied)|anxiety|anxieties|"
                  r"jitters|uncertainty|recession|ceasefire|truce|peace|deal|de-?escalat\w*|iran\w*|middle east|hormuz|"
                  r"houthi\w*|israel\w*|risk\w*|sell-?off|selloff|rout|crash\w*|plunge\w*|tumbl\w*|slump\w*|surg\w*|soar\w*|"
                  r"rall(?:y|ies|ied)|record|all-time|biggest|sharp\w*|\d+(?:\.\d+)?\s?%)\b")
EN_SAFE_HAVEN = (r"\b(?:safe[- ]havens? (?:demand|flows?|bid|buying|rush|assets? (?:rise|rally|gain|in demand))|flight to "
                 r"(?:safety|quality)|risk[- ]off|risk aversion|risk appetite (?:fades|sours|wanes|evaporates)|markets? "
                 r"(?:turmoil|panic|rout|meltdown)|global sell-?off|global selloff)\b")
EN_RISK_ON_WORDS = r"\b(?:risk[- ]on|risk appetite (?:returns|improves|recovers|revives)|relief rally)\b"
EN_EASING = re.compile(r"\b(?:tensions?|fears?|worries|concerns?|risks?|jitters)\s+(?:[\w\-]+\s+)?(?:eases?|eased|easing|recede\w*|"
                       r"fade\w*|abate\w*|cool\w*|subside\w*)|\b(?:easing|receding|fading|cooling) (?:of )?(?:[\w\-]+ )?"
                       r"(?:tensions?|fears?|worries|concerns?)|\bshrug\w* off\b|\bdespite\b|\bceasefire\b|\btruce\b|"
                       r"\bpeace (?:deal|talks|hopes)\b|\bde-?escalat\w*")
EN_INTERVENE = r"\b(?:interven\w+|rate checks?)\b"
# a new clause starts: the market move must come before it ("world shares are mixed and oil prices slip")
EN_CONJ = re.compile(r"\b(?:as|after|while|amid|despite|and|but|with|on|ahead of|before|following)\b")
# politicians asking for a policy move are not policy news ("Trump calls for Fed to cut rates")
EN_DEMAND = re.compile(r"\b(?:trump|white house|president|lawmakers?|politicians?|senators?|treasury secretary|bessent)\s+"
                       r"(?:[\w\-']+\s+){0,2}(?:calls?|urges?|urged|demands?|demanded|presses|pressed|pushes|pushed|wants|"
                       r"pressures?|pressured)\b")
EN_VERBAL = (r"\bwarns? (?:against|over|about)\b|\b(?:excessive|rapid|speculative|one-sided) (?:yen |currency |fx )?moves?\b|"
             r"\bclosely watching\b|\b(?:decisive|bold|appropriate) (?:forex |fx |currency )?(?:action|steps|measures)\b|"
             r"\b(?:ready|readiness) to act\b|\bstands? ready\b")
# Words that turn a policy cue around: "no rush to cut", "unlikely to hike", "pare bets on cuts", "cut bets fade".
EN_REVERSE_BEFORE = re.compile(r"\b(?:no rush to|not (?:in a )?(?:hurry|rush) to|unlikely to|won't|will not|no plans? to|rules? out|"
                               r"ruled out|push(?:es|ed)? back (?:on|against)|dismiss\w*|reject\w*|paus\w*|halt\w*|end(?:s|ed|ing)?|"
                               r"done with|scal\w* back|par(?:e|es|ed|ing)|trim\w*|dial\w* back|unwind\w*|fewer|less likely|slash\w*|"
                               r"abandon\w*|ditch\w*|price[sd]? out|pricing out|doubts? (?:over|about|on|of)|"
                               r"(?:cut\w*|lower\w*|reduc\w*) (?:the )?(?:odds|bets|chances|expectations|wagers|pricing|probability) "
                               r"(?:of|on|for))\s+(?:[\w\-]+\s+){0,5}$")
EN_REVERSE_AFTER = re.compile(r"[\s\-]*(?:[\w\-]+\s+){0,2}(?:bets?|expectations?|hopes?|odds|pricing|fears?)?\s*"
                              r"(?:fade[sd]?|recede[sd]?|wane[sd]?|dwindl\w*|diminish\w*|evaporat\w*|pared|trimmed|scaled back|unwound|"
                              r"off the table|priced out|ruled out|doubts?)")
EN_POLITICS = r"\b(?:election|snap poll|resign\w*|no-confidence|political (?:crisis|turmoil|uncertainty)|government collapse|impeach\w*)\b"
# a move verb does not move a currency when it is negated ("yen fails to rally", "dollar not falling")
EN_NEG_MOVE = re.compile(r"\b(?:fails? to|failed to|unable to|not|no longer|yet to|refuses? to|little|barely|hardly)\b|n't\b")
# a policy cue is turned around by a negation before it ("no rate cut", "not hawkish") ...
EN_NEG_CUE = re.compile(r"(?:\b(?:no|not|without|never|nor)|n't)\s+(?:[\w\-]+\s+){0,2}$")
# ... by its expectation fading ("rate hike odds fall") ...
EN_REVERSE_AFTER_FADE = re.compile(r"[\s\-]*(?:[\w\-]+\s+){0,2}(?:bets?|expectations?|hopes?|odds|pricing|fears?|chances?|wagers?)\s+"
                                   r"(?:fall|falls|fell|drop|drops|dropped|decline|declines|declined|ease|eases|eased|slip|slips|"
                                   r"shrink|shrinks|diminish|recede|recedes|fade|fades|faded|cool|cools|cooled|dim|dims|dimmed|"
                                   r"wane|wanes|waned|evaporate|evaporates)")
# ... or by a verb that deflates it ("jobs data douse rate hike bets")
EN_REVERSE_BEFORE_DEFLATE = re.compile(r"\b(?:douse[sd]?|dampen\w*|tempers?|tempered|curb\w*|dash\w*|cool(?:s|ed|ing)?|erode[sd]?|"
                                       r"dims?|dimmed|dimming|sap\w*|deflat\w*)\s+(?:[\w\-']+\s+){0,3}$")
# a risk-on word that is undone ("ceasefire with Iran is 'over'", "truce crumbles") is risk-off
EN_RISK_ON_UNDONE = re.compile(r"\b(?:ceasefire|truce|peace|deal)(?:[\s\W]+[\w'\-]+){0,4}?[\s\W]+(?:over|crumbl\w*|collaps\w*|ends?|"
                               r"ended|breaks? down|broke down|fails?|failed|falters?|violat\w*|in doubt|jeopardi\w*|shattered|"
                               r"unravel\w*)\b")
# a move verb only moves a currency named as a currency ("Australian property gains" and "US payroll gains"
# are not currency moves); "A rises against the dollar" moves the dollar the other way
EN_CUR_WORD = re.compile(r"(?:u\.?s\.? )?dollars?|greenback'?s?|usd|dxy|dollar index|yen|jpy|euros?|eur|pound|sterling|gbp|cable|"
                         r"aussie|australian dollar|aud")
EN_CLAUSE_BREAK = re.compile(r"[;:|]|\s[-–—]\s")
EN_AGAINST = re.compile(r"\b(?:against|versus|vs\.?)\s+(?:the\s+)?(?:u\.?s\.?\s+)?(dollar|greenback|yen|euro|pound|sterling|aussie)\b")
EN_AGAINST_CUR = {"dollar": "USD", "greenback": "USD", "yen": "JPY", "euro": "EUR", "pound": "GBP", "sterling": "GBP", "aussie": "AUD"}
# "dollar rises against the rupee", "dollar closes higher on Taipei forex market": a local emerging-market move
EN_EM_AFTER = re.compile(r"[^;:|]{0,40}?\b(?:against|versus|vs\.?|on|in)\s+(?:the\s+)?(?:[\w\-]+\s+){0,2}(?:rupees?|yuan|renminbi|won|"
                         r"pesos?|lira|ro?ubles?|ringgit|baht|rupiah|real|naira|rand|cedi|shillings?|dong|taka|dirham|riyal|"
                         r"zloty|forint|koruna|shekel|hryvnia|kwacha|tenge|taipei|mumbai|seoul|manila|jakarta|karachi|lagos|"
                         r"local)\b")

# ------------------------------------------------------------------ Japanese

JA_ENT = {
    "USD": ["米ドル", "米国", "米連邦", "FRB", "FOMC", "パウエル", "米金利", "米長期金利", "米国債", "米雇用", "米CPI",
            "米消費者物価", "米経済", "トランプ", "米財務", "米GDP", "米小売", "米"],
    "JPY": ["円相場", "円安", "円高", "円買い", "円売り", "円急落", "円急伸", "円急騰", "円反発", "円反落", "円続落",
            "円続伸", "円上昇", "円下落", "円キャリー", "日銀", "日本銀行", "植田", "為替介入", "財務省", "財務官", "財務相",
            "日本国債", "高市", "日本経済", "片山", "三村"],
    "EUR": ["ユーロ", "ECB", "欧州中銀", "欧州中央銀行", "ラガルド", "ユーロ圏", "ドイツ", "欧州"],
    "GBP": ["ポンド", "英国", "英中銀", "イングランド銀行", "BOE", "ベイリー", "英"],
    "AUD": ["豪ドル", "豪州", "オーストラリア", "豪中銀", "豪準備銀行", "RBA", "ブロック総裁"],
}
JA_MOVES = [  # longest first; (text, {currency: value})
    ("円安修正", {"JPY": 1}), ("円安是正", {"JPY": 1}), ("円安一服", {"JPY": 0.5}),
    ("円高修正", {"JPY": -1}), ("円高一服", {"JPY": -0.5}),
    ("ドル高一服", {"USD": -0.5}), ("ドル安一服", {"USD": 0.5}),
    ("豪ドル買い", {"AUD": 1}), ("豪ドル売り", {"AUD": -1}), ("ユーロ買い", {"EUR": 1}), ("ユーロ売り", {"EUR": -1}),
    ("ポンド買い", {"GBP": 1}), ("ポンド売り", {"GBP": -1}), ("ドル買い", {"USD": 1}), ("ドル売り", {"USD": -1}),
    ("豪ドル高", {"AUD": 1}), ("豪ドル安", {"AUD": -1}),
    ("ユーロ高", {"EUR": 1}), ("ユーロ安", {"EUR": -1}),
    ("ポンド高", {"GBP": 1}), ("ポンド安", {"GBP": -1}),
    ("ドル高", {"USD": 1}), ("ドル安", {"USD": -1}),
    ("円急落", {"JPY": -1}), ("円続落", {"JPY": -1}), ("円下落", {"JPY": -1}), ("円売り", {"JPY": -1}),
    ("円反落", {"JPY": -1}), ("円安", {"JPY": -1}),
    ("円急伸", {"JPY": 1}), ("円急騰", {"JPY": 1}), ("円反発", {"JPY": 1}), ("円上昇", {"JPY": 1}),
    ("円続伸", {"JPY": 1}), ("円買い", {"JPY": 1}), ("円高", {"JPY": 1}),
]
JA_PAIRS = [(f"{a}{sep}{b}", ca, cb)            # longest first: "豪ドル/米ドル" before "ドル/米ドル"
            for a, ca in (("豪ドル", "AUD"), ("ユーロ", "EUR"), ("ポンド", "GBP"), ("米ドル", "USD"), ("ドル", "USD"))
            for b, cb in (("米ドル", "USD"), ("ドル", "USD"), ("円", "JPY"))
            for sep in ("/", "・", "")
            if ca != cb]
JA_PAIRS.sort(key=lambda p: -len(p[0]))
# the direction of a pair: the last direction word before the next sentence or currency
# ("ドル円163円目前で急反落", "ドル/円今日の見通し｜GPIF巡る思惑で反落", "反発も依然として売りシグナル優勢")
JA_PAIR_DIR = re.compile(r"(?P<down>上昇分を削|伸び悩|上値(?:が|は)?重|下押し|下落|続落|急落|反落|軟調|弱含|割れ|売り優勢|"
                         r"売りシグナル|安値|下値を試|失速|戻り売り)|(?P<up>上昇|続伸|急伸|反発|上伸|急騰|高値更新|最高値|"
                         r"上値を試|底堅|堅調|強含|買い優勢|間近|目前|迫る|うかがう|上抜け)")
JA_PAIR_SCOPE_END = re.compile(r"。|ユーロ(?!圏)|ポンド|豪ドル|円相場|株|金利|原油")
# "ドルの戻りを阻む", "ドルの上値は重い", "ユーロ優位": a currency's direction in other words
JA_CUR_MOVE2 = re.compile(r"(豪ドル|ユーロ|ポンド|米ドル|ドル|円)(?:の)?(?:(?P<down>(?:戻り|上昇|反発)(?:を)?(?:阻|抑|鈍)|上値(?:が|は)?"
                          r"(?:重|限定)|(?:売り|下落)(?:が)?(?:優勢|加速))|(?P<up>下値(?:が|は)?(?:堅|固)|底堅|(?:が|は)?(?:優位|優勢)|"
                          r"(?:買い|上昇)(?:が)?(?:優勢|加速)))")
# "円安受け", "円安に対し": the weak yen as the reason for what follows, not a new move
JA_BACKDROP = re.compile(r"円安(?=(?:を)?受け|に対し|への対応|を警戒|警戒|けん制|牽制)|円高(?=(?:を)?受け|に対し|への対応)")
# signals said to be mixed: "利上げ観測と介入警戒が交錯", "高値圏の攻防"
JA_MIXED = re.compile(r"交錯|攻防|綱引き|拮抗|もみ合い|方向感")
JA_CUR_MOVE = re.compile(r"(豪ドル|ユーロ|ポンド|ドル|円)(?:相場)?(?:は|が|も)\s?"
                         r"(?:(?P<down>急落|続落|下落|反落|軟調|売られ|弱含)|(?P<up>急伸|急騰|続伸|上昇|反発|堅調|買われ|強含))")
JA_CUR_CODE = {"豪ドル": "AUD", "ユーロ": "EUR", "ポンド": "GBP", "ドル": "USD", "円": "JPY"}
JA_YEN_RATE = re.compile(r"円相場[^。]{0,16}?(?:(?P<down>急落|続落|下落|反落)|(?P<up>急伸|急騰|続伸|上昇|反発))")
JA_NOT_MOVE = re.compile(r"(?:に|へ|と|は)?(?:つながらず|つながらない|ならず|ならない|至らず|限定的|見込めず|進まず)")
JA_PAIR_UP = re.compile(r"^(?:相場)?(?:が|は)?\s?(?:上昇|続伸|急伸|反発|高|上伸)")
JA_PAIR_DOWN = re.compile(r"^(?:相場)?(?:が|は)?\s?(?:下落|続落|急落|反落|安|下押し)")
JA_HAWK = (r"利上げ|引き締め|タカ派|インフレ加速|物価上昇|物価高|金利を?[^、。]{0,8}引き上げ|"
           r"(?:物価|インフレ率?|CPI|消費者物価)[^。]{0,10}?(?<!緩やかな)(?<!小幅な)(?<!わずかな)(?:上昇|加速|上振れ|高水準)")
JA_DOVE = (r"利下げ|(?:金融|追加|量的)緩和|緩和(?:策|的|姿勢|継続|方針|政策)|ハト派|景気後退|景気減速|減速|物価下落|"
           r"金利を?[^、。]{0,8}引き下げ|"
           r"(?:物価|インフレ率?|CPI|消費者物価)[^、。]{0,8}?(?:鈍化|減速|低下|下振れ)")
JA_BEAT = r"予想(?:を)?(?:大幅に|大きく|やや|わずかに)?上回|上振れ|好調|改善"
JA_MISS = r"予想(?:を)?(?:大幅に|大きく|やや|わずかに)?下回|下振れ|悪化|低迷"
JA_REVERSE = re.compile(r"(?:観測|期待|予想|見通し|機運|織り込み|姿勢)?(?:の|が|は|を|も)?\s?"
                        r"(?:僅かに|わずかに|やや|大きく|急速に|一段と|さらに)?"
                        r"(?:後退|遠の|剥落|はく落|見送|急が|慎重|否定|打ち消|織り込み過ぎ|せず|しない|休止|停止|終了|縮小|解除|出口|修正)"
                        r"|[」「はが、\s]*(?:それほど|さほど|あまり)?(?:強くない|高くない|大きくない|限定的|一時的)")
JA_YIELD_UP = r"米(?:国)?の?(?:長期)?金利(?:が|は)?[^、。]{0,6}(?:上昇|高水準|最高)|米国債利回り(?:が|は)?[^、。]{0,4}上昇"
JA_YIELD_DOWN = r"米(?:国)?の?(?:長期)?金利(?:が|は)?[^、。]{0,6}(?:低下|下落)|米国債利回り(?:が|は)?[^、。]{0,4}低下"
# risk, as in English: explicit risk-off words, or a broad stock move with a risk reason
JA_RISK_EXPLICIT_OFF = re.compile(r"リスクオフ|リスク回避|安全資産|有事の円買い|質への逃避|世界同時株安")
JA_RISK_EXPLICIT_ON = re.compile(r"リスクオン|リスク選好")
JA_STOCKS = re.compile(r"株価|株式|株式市場|日経平均|日経株価|東証|ダウ|NY株|米国株|米株|世界株|アジア株|欧州株|S&P|ナスダック")
JA_STOCK_MOVE = re.compile(r"(?P<down>急落|暴落|大幅安|続落|下落|反落|株安)|(?P<up>急騰|大幅高|続伸|上昇|反発|最高値|株高)")
JA_RISK_REASON = re.compile(r"地政学|紛争|戦争|攻撃|ミサイル|緊張|制裁|関税|懸念|不安|停戦|合意|和平|急|大幅|暴")
JA_INTERVENE = r"為替介入|介入|レートチェック"
JA_VERBAL = r"けん制|牽制|過度な変動|投機的な動き|憂慮|断固たる(?:対応|措置)|断固とした|あらゆる(?:手段|措置)|緊張感を持って"
JA_POLITICS = r"解散|総選挙|辞任|政局|不信任"

RISK_OFF_EFFECT = {"JPY": 0.4, "USD": 0.15, "AUD": -0.4}
RISK_ON_EFFECT = {"JPY": -0.3, "AUD": 0.3}
_OTHER_DOLLARS = ("豪", "NZ", "加", "カナダ", "香港", "シンガポール", "台湾")


def _keep_entity(t: str, title: str, s: int, e: int, fx_ctx: bool, mon_ctx: bool) -> bool:
    """Ambiguous names count only with a matching context."""
    w = t[s:e]
    if w in _NEEDS_FX or w == "pound":
        if w == "pound" and _NUMBER_BEFORE.search(t[max(0, s - 12):s]):
            return False                                   # "5,000-pound", "145 pound"
        rest = t[s:].replace(w, "", 1)
        return bool(EN_FX_CTX.search(rest) or EN_FX_CTX.search(t[:s]) or EN_VERB.match(rest.lstrip()))
    if w in _NEEDS_MON:
        return mon_ctx and bool(EN_MON_CTX.search(t[:s] + " " + t[e:]))
    if w == "us":
        return title[s:e] == "US"                         # the pronoun "us" is not the United States
    return True


def _entities_en(t: str, title: str | None = None) -> list[tuple[int, int, str]]:
    title = t if title is None else title
    fx_ctx = bool(EN_FX_CTX.search(t))
    mon_ctx = bool(EN_MON_CTX.search(t))
    ents = []
    for cur, pats in EN_ENT.items():
        for p in pats:
            for m in re.finditer(p, t):
                if cur in CURRENCIES and not _keep_entity(t, title, m.start(), m.end(), fx_ctx, mon_ctx):
                    continue
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


def _nearest(ents, pos: int):
    """The tracked currency nearest to ``pos``; None when another country or central bank is nearer."""
    best, dist = None, math.inf
    for s, e, cur in ents:
        if cur not in TRACKED_OR_ELSEWHERE:
            continue
        d = 0 if s <= pos < e else min(abs(pos - s), abs(pos - e))
        if d < dist:
            best, dist = cur, d
    return best if best in CURRENCIES else None


def _is_rate_hike(t: str, m: re.Match) -> bool:
    """A bare "hike" is a rate hike with a central bank or "rate" in the headline and not "price hikes"."""
    before = t[max(0, m.start() - 25):m.start()].split()[-2:]
    after = t[m.end():m.end() + 25].split()[:2]
    if any(EN_HIKE_NOT.fullmatch(w.strip(",.:;'\"")) for w in before + after):
        return False
    return bool(EN_CB.search(t))


def _risk_en(t: str) -> int:
    """+1 risk-on, -1 risk-off, 0 neither."""
    broad = [m for m in EN_BROAD.finditer(t)]
    if broad and EN_LOCAL_MARKET.search(t) and not EN_MAJOR_MARKET.search(t):
        broad = []                                          # "Sensex slides", "Seoul shares rise": local markets
    d = 0
    for b in broad:
        m = EN_MKT_MOVE.search(t, b.end(), b.end() + 30)
        if m and not EN_CLAUSE_BREAK.search(t, b.end(), m.start()) and not EN_NEG_MOVE.search(t[b.end():m.start()]) \
                and not EN_CONJ.search(t[b.end():m.start()]):
            d = 1 if m.group("up") else -1
            break
    if d and re.search(EN_RISK_REASON, t):
        if d < 0 and EN_EASING.search(t) and not EN_RISK_ON_UNDONE.search(t):
            return 0                                        # "stocks slip despite ceasefire": mixed
        return d
    if re.search(EN_SAFE_HAVEN, t):
        return -1
    if broad and re.search(EN_RISK_ON_WORDS, t):
        return 1
    if broad and EN_RISK_ON_UNDONE.search(t):
        return -1                                           # "stocks drop as ceasefire ends"
    return 0


def _analyze_en(title: str, default_cur: str | None, add, effects: dict) -> tuple:
    t = title.lower()
    moved: set[str] = set()

    def effects_sign(cur):
        return effects.get(cur, 0.0)

    def move(cur, v):
        if cur in CURRENCIES and v:
            add(cur, v, "fx_move")
            moved.add(cur)

    # pairs: "USD/JPY", "EUR/AUD", "EURUSD", "Pound to Euro", "Pound Euro"
    spans = []
    for p in EN_PAIR.finditer(t):
        a, b = p.group(1), p.group(2)
        if a == b:
            continue
        spans.append((p.start(), p.end(), a.upper(), b.upper(), all(x in EN_G10.split("|") for x in (a, b))))
    for p in EN_WORD_PAIR.finditer(t):
        a, b = EN_WORD_CUR[p.group(1)], EN_WORD_CUR[p.group(2)]
        if a != b and not any(s <= p.start() < e for s, e, *_ in spans):
            nxt = t[p.end():p.end() + 1]
            if nxt in ("", " ", ":", ",", "-") and (EN_PAIR_VERB.match(t, p.end(), p.end() + 45)
                                                    or EN_PAIR_LABEL.match(t, p.end())):
                spans.append((p.start(), p.end(), a, b, True))
    for s, e, base, quote, major in spans:
        if not major:
            continue                                        # "USD/CNY dips": says little about the majors
        m = EN_PAIR_VERB.match(t, e, e + 45)
        if not m:
            m2 = EN_VIEW.search(t, e, e + 60)                # "EUR/AUD bulls hold the higher ground"
            if not m2 or EN_CLAUSE_BREAK.search(t, e, m2.start()):
                m2 = EN_NOUN_BEFORE.search(t[max(0, s - 30):s])   # "a recovery in EUR/USD"
                if not m2:
                    continue
            m = m2
        d = 1 if m.group("up") else -1
        move(base, d)
        move(quote, -d)
    ents = [x for x in _entities_en(t, title) if not any(s <= x[0] < e for s, e, *_ in spans)]
    for s, e, cur in ents:
        if cur in CURRENCIES and not EN_CUR_WORD.fullmatch(t[s:e]):
            continue                                        # "Fed", "Australian", "US": not the currency itself
        if cur == "AUD" and t[s:e] == "aussie" and not (EN_FX_CTX.search(t[:s] + t[e:]) or EN_VERB.match(t, e + 1)):
            continue                                        # "Aussie shares", "Aussie business"
        d = 0
        m = EN_VERB.search(t, e, e + 40)
        if m and EN_CLAUSE_BREAK.search(t, e, m.start()):
            m = None                                        # "yen range; falling yields": another clause's verb
        if m and m.start() - e <= 25 and not any(e <= s2 < m.start() for s2, _, _ in ents):
            d = 1 if m.group("up") else -1
            if EN_NEG_MOVE.search(t[e:m.start()]) or EN_OTHER_SUBJECT.search(t[e:m.start()]):
                d = 0                                       # "yen fails to rally", "yen intervention weakens dollar"
        if not d and (m := EN_NOUN_AFTER.match(t, e)) and m.end() - e <= 40 and not EN_CLAUSE_BREAK.search(t, e, m.start()):
            d = 1 if m.group("up") else -1                  # "USD selloff accelerates", "persistent weakness"
        if not d and (m := EN_NOUN_BEFORE.search(t[max(0, s - 30):s])):
            d = 1 if m.group("up") else -1                  # "positive on the USD"
        before = t[max(0, s - 14):s]
        if re.search(r"\b(?:stronger|firmer)\s+(?:the\s+)?(?:u\.?s\.?\s+)?$", before):
            d = 1
        elif re.search(r"\b(?:weaker|softer)\s+(?:the\s+)?(?:u\.?s\.?\s+)?$", before):
            d = -1
        if re.search(EN_OBJ_DOWN + r"(?:u\.?s\.?\s+)?$", t[max(0, s - 24):s]):
            d = -1
        elif re.search(EN_OBJ_UP + r"(?:u\.?s\.?\s+)?$", t[max(0, s - 24):s]):
            d = 1
        if not d or cur not in CURRENCIES:
            continue                    # "rupee falls against the dollar" says little about the majors
        if cur == "USD" and EN_EM_AFTER.match(t, e):
            continue                    # "dollar closes higher on Taipei forex market"
        move(cur, d)
        if (a := EN_AGAINST.search(t, e, e + 70)) and EN_AGAINST_CUR[a.group(1)] != cur:
            move(EN_AGAINST_CUR[a.group(1)], -d)             # "yen jumps against the dollar"
    for s, e, base, quote, major in spans:                  # "USD/JPY forecast: yen keeps showing weakness"
        if major and (base in moved) != (quote in moved) and {base, quote} <= set(CURRENCIES):
            one, other = (base, quote) if base in moved else (quote, base)
            move(other, -1 if effects_sign(one) > 0 else 1)

    def reversed_at(m):
        before = t[max(0, m.start() - 60):m.start()]
        return bool(EN_REVERSE_BEFORE.search(before) or EN_REVERSE_AFTER.match(t, m.end())
                    or EN_NEG_CUE.search(t[max(0, m.start() - 30):m.start()]) or EN_REVERSE_AFTER_FADE.match(t, m.end())
                    or EN_REVERSE_BEFORE_DEFLATE.search(before))

    questions = [(q.start(), q.end()) for q in EN_QUESTION.finditer(t)]
    cue_hits = []
    for pattern, value in ((EN_HAWK, 0.8), (EN_DOVE, -0.8), (EN_CB_CUT, -0.8)):
        cue_hits += [(m, value, "policy") for m in re.finditer(pattern, t)]
    covered = [(m.start(), m.end()) for m, _, _ in cue_hits]
    for m in EN_HIKE_WORD.finditer(t):
        if not any(s <= m.start() < e for s, e in covered) and _is_rate_hike(t, m):
            cue_hits.append((m, 0.8, "policy"))
    for pattern, value in ((EN_BEAT, 0.6), (EN_MISS, -0.6)):
        for m in re.finditer(pattern, t):
            if re.search(r"\b" + EN_MACRO + r"\b", t[max(0, m.start() - 40):m.start()]):
                cue_hits.append((m, value, "data"))          # "payrolls beat forecasts", not "ASML tops forecasts"
    for pattern, value in ((EN_DATA_UP, 0.6), (EN_DATA_DOWN, -0.6), (EN_DATA_UP_MORE, 0.6)):
        cue_hits += [(m, value, "data") for m in pattern.finditer(t)
                     if not any(s <= m.start() < e for s, e in covered)]
    for m, value, topic in cue_hits:
        if topic == "policy" and (any(a <= m.start() < b for a, b in questions) or EN_DEMAND.search(t[:m.start()])):
            continue                                        # "Will the Fed raise rates?", "Trump urges Fed to cut"
        cur = _nearest(ents, m.start()) or default_cur
        if topic == "policy" and reversed_at(m):
            value = -value
        add(cur, value, topic)
    if any(EN_CUR_WORD.fullmatch(t[s:e]) for s, e, c in ents if c in CURRENCIES):
        for m in re.finditer(EN_POLITICS, t):              # politics only in a currency headline
            add(_nearest(ents, m.start()) or default_cur, -0.4, "politics")
    us_yields = bool(EN_YIELD_US_CTX.search(t)) or not EN_YIELD_NOT_US.search(t)
    if us_yields and re.search(EN_YIELD_UP, t):
        add("USD", 0.6, "yields")
    if us_yields and re.search(EN_YIELD_DOWN, t):
        add("USD", -0.6, "yields")
    r = _risk_en(t)
    for cur, v in (RISK_OFF_EFFECT if r < 0 else RISK_ON_EFFECT if r > 0 else {}).items():
        if cur not in moved:                                # a currency's own reported move comes first
            add(cur, v, "risk")
    pair_legs = [(s, e, c) for s, e, base, quote, major in spans if major for c in (base, quote)]
    return ents + pair_legs, moved, t


def analyze_lexicon(title: str, lang: str, default_cur: str | None = None) -> dict:
    """Keyword analysis of one headline -> {"by", "cur": {currency: score}, "men": [...], "top": [...]}."""
    effects: dict[str, float] = {}
    topics: set[str] = set()

    def add(cur, v, topic):
        if cur in CURRENCIES and v:
            effects[cur] = effects.get(cur, 0.0) + v
            topics.add(topic)

    if lang == "ja":
        t = unicodedata.normalize("NFKC", title)
        ents = _entities_ja(t)
        moved = set()
        masked = JA_BACKDROP.sub(lambda m: "＿" * len(m.group(0)), t)
        for word, eff in JA_MOVES:
            while (i := masked.find(word)) >= 0:
                if not JA_NOT_MOVE.match(t, i + len(word)):      # "円高につながらず" is not a yen rise
                    for cur, v in eff.items():
                        add(cur, v, "fx_move")
                        moved.add(cur)
                masked = masked[:i] + "＿" * len(word) + masked[i + len(word):]
        # Pair names ("ドル円", "米ドル/円") are scored as pairs below, not as their component currencies.
        single = masked
        pair_at = []
        for word, base, quote in JA_PAIRS:
            while (i := single.find(word)) >= 0:
                pair_at.append((i, i + len(word), base, quote))
                single = single[:i] + "＿" * len(word) + single[i + len(word):]
        for m in list(JA_CUR_MOVE.finditer(single)) + list(JA_CUR_MOVE2.finditer(single)):
            cur = JA_CUR_CODE.get(m.group(1), "USD")
            if cur == "USD" and m.start() > 0 and single[m.start() - 1] in "豪加":
                continue
            add(cur, 1 if m.group("up") else -1, "fx_move")
            moved.add(cur)
        for m in JA_YEN_RATE.finditer(single):
            add("JPY", 1 if m.group("up") else -1, "fx_move")
            moved.add("JPY")
        for i, j, base, quote in sorted(pair_at):
            end = JA_PAIR_SCOPE_END.search(single, j, j + 40)
            scope = t[j:end.start() if end else j + 40]
            d = 0
            if JA_PAIR_UP.match(scope):
                d = 1
            elif JA_PAIR_DOWN.match(scope):
                d = -1
            else:
                dirs = list(JA_PAIR_DIR.finditer(scope))
                if dirs:
                    d = 1 if dirs[-1].group("up") else -1
            if d:
                add(base, d, "fx_move")
                add(quote, -d, "fx_move")
                moved |= {base, quote}
        mixed = bool(JA_MIXED.search(t))
        cues = [(JA_HAWK, 0.8, "policy"), (JA_DOVE, -0.8, "policy"), (JA_BEAT, 0.6, "data"),
                (JA_MISS, -0.6, "data"), (JA_POLITICS, -0.3, "politics")]
        for pattern, value, topic in cues:
            if mixed:
                break                                    # "利上げ観測と介入警戒が交錯": no direction
            for m in re.finditer(pattern, t):
                cur = _nearest(ents, m.start()) or default_cur
                value_m = -value if topic == "policy" and JA_REVERSE.match(t, m.end()) else value
                add(cur, value_m, topic)
        if re.search(JA_YIELD_UP, t):
            add("USD", 0.6, "yields")
        if re.search(JA_YIELD_DOWN, t):
            add("USD", -0.6, "yields")
        r = 0
        if JA_RISK_EXPLICIT_OFF.search(t):
            r = -1
        elif JA_RISK_EXPLICIT_ON.search(t):
            r = 1
        elif (st := JA_STOCKS.search(t)) and (mv := JA_STOCK_MOVE.search(t, st.end(), st.end() + 12)) and JA_RISK_REASON.search(t):
            r = 1 if mv.group("up") else -1
        for cur, v in (RISK_OFF_EFFECT if r < 0 else RISK_ON_EFFECT if r > 0 else {}).items():
            if cur not in moved:
                add(cur, v, "risk")
        ents = ents + [(i, j, c) for i, j, base, quote in pair_at for c in (base, quote)]
        if mixed:
            moved.add("JPY")                             # no intervention call either
        intervene, verbal = JA_INTERVENE, JA_VERBAL
    else:
        ents, moved, t = _analyze_en(title, default_cur, add, effects)
        intervene, verbal = EN_INTERVENE, EN_VERBAL
    mentioned = sorted({c for _, _, c in ents if c in CURRENCIES} | ({default_cur} if default_cur else set()))
    # An intervention mention supports the yen unless the headline already says how the yen moved.
    if re.search(intervene, t) and "JPY" in mentioned and "JPY" not in moved:
        add("JPY", 0.8, "intervention")
    elif re.search(verbal, t) and "JPY" in mentioned and "JPY" not in moved:
        add("JPY", 0.5, "intervention")    # verbal warnings against a weak yen
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


_SUFFIX = re.compile(r"(?:執筆[：:]\s*\S+|[（(][^）)]{1,20}[）)]|[-|｜].{1,30})\s*$")
STORY_WINDOW = timedelta(hours=12)
STORY_SIMILARITY = 0.6


def _shingles(title: str) -> frozenset:
    t = _SUFFIX.sub("", title)
    t = re.sub(r"[\W_]+", "", t.lower())
    return frozenset(t[i:i + 3] for i in range(max(1, len(t) - 2)))


def stories(items: list[dict]) -> dict[str, int]:
    """Group near-identical headlines (the same article syndicated on several sites) into stories.

    Returns item id -> story number. Deterministic for a given set of items.
    """
    out: dict[str, int] = {}
    reps: list[tuple[datetime, frozenset, int]] = []
    for it in sorted(items, key=lambda x: (x["published_at"], x["id"])):
        when = parse_iso(it["published_at"])
        sh = _shingles(it["title"])
        found = None
        for t0, sh0, sid in reversed(reps):
            if when - t0 > STORY_WINDOW:
                break
            if len(sh & sh0) / max(1, len(sh | sh0)) >= STORY_SIMILARITY:
                found = sid
                break
        if found is None:
            found = len(reps)
            reps.append((when, sh, found))
        out[it["id"]] = found
    return out


def pressures(items: list[dict], cutoff: datetime) -> dict[str, dict]:
    """News pressure per currency: recency- and source-weighted mean score, shrunk toward 0.

    Copies of the same story share one story's weight, so an article
    syndicated on five sites does not count five times.
    """
    num = {c: 0.0 for c in CURRENCIES}
    den = {c: 0.0 for c in CURRENCIES}
    cnt = {c: set() for c in CURRENCIES}
    elig = eligible(items, cutoff)
    story = stories(elig)
    copies: dict[int, int] = {}
    for sid in story.values():
        copies[sid] = copies.get(sid, 0) + 1
    for it in elig:
        age_h = (cutoff - parse_iso(it["published_at"])).total_seconds() / 3600.0
        sid = story[it["id"]]
        d = SOURCE_WEIGHT.get(it["src"], 0.8) * math.exp(-age_h / TAU_HOURS) / copies[sid]
        for cur, score in it["an"]["cur"].items():
            num[cur] += d * score
            den[cur] += d
            cnt[cur].add(sid)
    return {c: {"p": round(num[c] / (SHRINK + den[c]), 6), "w": round(den[c], 4), "n": len(cnt[c])} for c in CURRENCIES}


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
