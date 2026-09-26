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

ANALYZER = "lexicon-v3"
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
            r"\btreasur(?:y|ies)\b", r"\bnonfarm\b", r"\bpayrolls\b", r"\bu\.?s\.?\b", r"\bamerica\b", r"\btrump\b",
            r"\bwarsh\b", r"\bbessent\b", r"\bgreenback'?s?\b"],
    "JPY": [r"\byen\b", r"\bjpy\b", r"\bbank of japan\b", r"\bboj\b", r"\bueda\b", r"\bjapan(?:ese|'s)?\b", r"\bjgbs?\b",
            r"\btakaichi\b", r"\bkatayama\b", r"\bmimura\b"],
    "EUR": [r"\beuros?\b(?!pe)", r"\beur\b", r"\becb\b", r"\blagarde\b", r"\beuro ?zone\b", r"\beuro area\b", r"\bbunds?\b",
            r"\bgerman(?:y|'s)?\b", r"\bfrance\b", r"\bfrench\b"],
    "GBP": [r"\bpound\b", r"\bsterling\b", r"\bgbp\b", r"\bcable\b", r"\bbank of england\b", r"\bboe\b", r"\bbailey\b",
            r"\bgilts?\b", r"\bu\.?k\.?\b", r"\bbritain\b", r"\bbritish\b", r"\breeves\b"],
    "AUD": [r"\baussie\b", r"\baustralian dollar\b", r"\baud\b", r"\brba\b", r"\breserve bank of australia\b",
            r"\bbullock\b", r"\baustralia(?:n|'s)?\b", r"\bchalmers\b"],
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
EN_UP_MORE = r"edges? (?:up|higher)|inch(?:es|ed)? (?:up|higher)|ticks? (?:up|higher)|steadies|recoups|holds gains"
EN_DOWN_MORE = (r"eases|eased|softens|softened|edges? (?:down|lower)|inch(?:es|ed)? (?:down|lower)|ticks? (?:down|lower)|"
                r"pares gains|gives up gains|(?:hovers|holds|trades|stays) (?:near|around|at) [\w\- ]{0,15}lows?")
EN_VERB = re.compile(r"\b(?:(?P<up>" + EN_UP_MORE + "|" + EN_UP + r")|(?P<down>" + EN_DOWN_MORE + "|" + EN_DOWN + r"))")
EN_PAIR_VERB = re.compile(r"[\s:,\-]*(?:(?:rate|pair|exchange rate|price)\s+)?(?:(?P<up>" + EN_UP + r")|(?P<down>" + EN_DOWN + r"))")
EN_HAWK = r"\b(?:hikes?|hiking|hiked|raises? rates|rate (?:rise|increase|hike)s?|tighten(?:s|ing)?|hawkish|higher for longer|inflation (?:rises|jumps|accelerates|surges|heats|hotter)|sticky inflation)\b"
EN_DOVE = r"\b(?:rate cuts?|cuts? rates|cutting|lowers? rates|easing|eases policy|dovish|stimulus|recession|slowdown|inflation (?:cools|eases|slows|falls)|disinflation)\b"
EN_BEAT = r"\b(?:beats?|tops) (?:expectations|estimates|forecasts)\b|\b(?:better|stronger)[- ]than[- ]expected\b|\babove (?:expectations|forecasts|estimates)\b"
EN_MISS = r"\bmiss(?:es|ed)? (?:expectations|estimates|forecasts)\b|\b(?:worse|weaker|softer)[- ]than[- ]expected\b|\bbelow (?:expectations|forecasts|estimates)\b"
EN_YIELD_UP = (r"\b(?:treasury|u\.?s\.?|us|10-year|bond) yields? (?:(?:are|were|is|keep|kept) )?(?:[\w\-]+ing )?"
               r"(?:rise|rises|rose|jump|jumps|jumped|climb|climbs|surge|surges|surged|soar|soars|higher|spike|spikes|hit)"
               r"|\b(?:higher|rising|surging|soaring) (?:u\.?s\.? |us |treasury |bond |10-year )?yields\b")
EN_YIELD_DOWN = (r"\b(?:treasury|u\.?s\.?|us|10-year|bond) yields? (?:(?:are|were|is|keep|kept) )?(?:[\w\-]+ing )?"
                 r"(?:fall|falls|fell|drop|drops|dropped|slide|slides|lower|tumble|tumbles|ease|eases)"
                 r"|\b(?:lower|falling|sliding|tumbling) (?:u\.?s\.? |us |treasury |bond |10-year )?yields\b")
EN_RISK_OFF = r"\b(?:war|attacks?|missiles?|airstrikes?|conflict|invasion|sanctions|crisis|turmoil|sell-?off|crash|panic|safe[- ]haven|geopolitic\w*|tensions|escalat\w*|trade war|shutdown|default)\b"
EN_RISK_ON = r"\b(?:stocks? rally|risk appetite|risk-on|ceasefire|truce|trade deal|deal reached)\b"
EN_INTERVENE = r"\b(?:interven\w+|rate checks?)\b"
EN_VERBAL = r"\bwarns? (?:against|over|about)\b|\b(?:excessive|rapid|speculative|one-sided) (?:yen |currency |fx )?moves?\b|\bclosely watching\b"
# Words that turn a policy cue around: "no rush to cut", "unlikely to hike", "pare bets on cuts", "cut bets fade".
EN_REVERSE_BEFORE = re.compile(r"\b(?:no rush to|not (?:in a )?(?:hurry|rush) to|unlikely to|won't|will not|no plans? to|rules? out|ruled out|"
                               r"push(?:es|ed)? back (?:on|against)|dismiss\w*|reject\w*|paus\w*|halt\w*|end(?:s|ed|ing)?|done with|"
                               r"scal\w* back|par(?:e|es|ed|ing)|trim\w*|dial\w* back|unwind\w*|fewer|less likely)\s+(?:[\w\-]+\s+){0,3}$")
EN_REVERSE_AFTER = re.compile(r"[\s\-]*(?:[\w\-]+\s+){0,2}(?:bets?|expectations?|hopes?|odds|pricing|fears?)?\s*"
                              r"(?:fade[sd]?|recede[sd]?|wane[sd]?|dwindl\w*|diminish\w*|evaporat\w*|pared|trimmed|scaled back|unwound|"
                              r"off the table|priced out|ruled out)")
EN_POLITICS = r"\b(?:election|snap poll|resign\w*|no-confidence|political (?:crisis|turmoil|uncertainty)|government collapse|impeach\w*)\b"
# Fixes checked on hand-read GDELT headlines (research/news.md: correct readings 11 -> 20 of 30 on the
# tune sample, 9 -> 18 of 30 on the test sample):
# a move verb does not move a currency when it is negated ("yen fails to rally", "dollar not falling")
EN_NEG_MOVE = re.compile(r"\b(?:fails? to|failed to|struggles? to|struggled to|unable to|not|no longer|yet to|"
                         r"refuses? to|little|barely|hardly)\b|n't\b")
# a policy cue is turned around by a negation before it ("no rate cut", "not hawkish") ...
EN_NEG_CUE = re.compile(r"(?:\b(?:no|not|without|never|nor)|n't)\s+(?:[\w\-]+\s+){0,2}$")
# ... by its expectation fading ("rate hike odds fall") ...
EN_REVERSE_AFTER_FADE = re.compile(r"[\s\-]*(?:[\w\-]+\s+){0,2}(?:bets?|expectations?|hopes?|odds|pricing|fears?|chances?)\s+"
                                   r"(?:fall|falls|fell|drop|drops|dropped|decline|declines|declined|ease|eases|eased|slip|slips|"
                                   r"shrink|shrinks|diminish|recede|recedes|fade|fades|faded|cool|cools)")
# ... or by a verb that deflates it ("jobs data douse rate hike bets")
EN_REVERSE_BEFORE_DEFLATE = re.compile(r"\b(?:douse[sd]?|dampen\w*|tempers?|tempered|curb\w*|dash\w*|cool(?:s|ed|ing)?|erode[sd]?)\s+"
                                       r"(?:[\w\-']+\s+){0,3}$")
# a risk-on word that is undone ("ceasefire with Iran is 'over'", "truce crumbles") is risk-off
EN_RISK_ON_UNDONE = re.compile(r"(?:[\s\W]+[\w'\-]+){0,4}?[\s\W]+(?:over|crumbl\w*|collaps\w*|ends?|ended|breaks? down|broke down|"
                               r"fails?|failed|falters?|violat\w*|in doubt|jeopardi\w*|shattered|unravel\w*)\b")
# a move verb only moves a currency named as a currency ("Australian property gains" and "US payroll gains"
# are not currency moves); "A rises against the dollar" moves the dollar the other way
EN_CUR_WORD = re.compile(r"(?:u\.?s\.? )?dollars?|greenback'?s?|usd|dxy|yen|jpy|euros?|eur|pound|sterling|gbp|cable|aussie|"
                         r"australian dollar|aud")
EN_CLAUSE_BREAK = re.compile(r"[;:|]|\s[-–—]\s")
EN_AGAINST = re.compile(r"\b(?:against|versus|vs\.?)\s+(?:the\s+)?(?:u\.?s\.?\s+)?(dollar|greenback|yen|euro|pound|sterling|aussie)\b")
EN_AGAINST_CUR = {"dollar": "USD", "greenback": "USD", "yen": "JPY", "euro": "EUR", "pound": "GBP", "sterling": "GBP", "aussie": "AUD"}
# risk-off and risk-on words count only in a market headline (war and politics stories are not market news)
EN_MARKET_CTX = re.compile(r"\b(?:markets?|stocks?|shares|equities|investors|traders|safe[- ]haven|risk[- ](?:off|on|appetite|"
                           r"sentiment|aversion|assets)|currenc\w*|forex|fx|yields?|bonds?|treasur\w*|dollar|yen|euro|sterling|"
                           r"pound|aussie|gold|oil|wall street|nikkei|s&p|ftse|dow|nasdaq|dax|sensex|nifty|hang seng|asx|"
                           r"kospi|stoxx)\b")
JA_MARKET_CTX = re.compile(r"株|相場|市場|円|ドル|ユーロ|ポンド|金利|為替|投資家|リスク")

JA_ENT = {
    "USD": ["米ドル", "米国", "米連邦", "FRB", "FOMC", "パウエル", "米金利", "米長期金利", "米国債", "米雇用", "米CPI",
            "米消費者物価", "米経済", "トランプ", "米財務", "米GDP", "米小売", "米"],
    "JPY": ["円相場", "円安", "円高", "円買い", "円売り", "円急落", "円急伸", "円急騰", "円反発", "円反落", "円続落",
            "円続伸", "円上昇", "円下落", "円キャリー", "日銀", "日本銀行", "植田", "為替介入", "財務省", "財務官", "財務相",
            "日本国債", "高市", "日本経済"],
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
JA_PAIRS = [("豪ドル円", "AUD", "JPY"), ("ユーロ円", "EUR", "JPY"), ("ポンド円", "GBP", "JPY"),
            ("ユーロドル", "EUR", "USD"), ("ポンドドル", "GBP", "USD"), ("豪ドル米ドル", "AUD", "USD"),
            ("ドル円", "USD", "JPY"), ("ドル・円", "USD", "JPY")]
# "円は下落", "ユーロが上昇": currency, optional particle, then the move.
JA_CUR_MOVE = re.compile(r"(豪ドル|ユーロ|ポンド|ドル|円)(?:相場)?(?:は|が|も)\s?"
                         r"(?:(?P<down>急落|続落|下落|反落|軟調|売られ|弱含)|(?P<up>急伸|急騰|続伸|上昇|反発|堅調|買われ|強含))")
JA_CUR_CODE = {"豪ドル": "AUD", "ユーロ": "EUR", "ポンド": "GBP", "ドル": "USD", "円": "JPY"}
# "円相場、1ドル=160円台に下落": the yen rate moved, even with words in between.
JA_YEN_RATE = re.compile(r"円相場[^。]{0,16}?(?:(?P<down>急落|続落|下落|反落)|(?P<up>急伸|急騰|続伸|上昇|反発))")
JA_NOT_MOVE = re.compile(r"(?:に|へ|と|は)?(?:つながらず|つながらない|ならず|ならない|至らず|限定的|見込めず|進まず)")
JA_PAIR_UP = re.compile(r"^(?:相場)?(?:が|は)?\s?(?:上昇|続伸|急伸|反発|高|上伸)")
JA_PAIR_DOWN = re.compile(r"^(?:相場)?(?:が|は)?\s?(?:下落|続落|急落|反落|安|下押し)")
JA_HAWK = r"利上げ|引き締め|タカ派|インフレ加速|物価上昇|物価高"
JA_DOVE = r"利下げ|金融緩和|緩和|ハト派|景気後退|景気減速|減速|物価下落"
JA_BEAT = r"予想(?:を)?(?:大幅に|大きく|やや|わずかに)?上回|上振れ|好調|改善"
JA_MISS = r"予想(?:を)?(?:大幅に|大きく|やや|わずかに)?下回|下振れ|悪化|低迷"
# A policy cue followed by one of these means the opposite: "利上げ観測が後退", "利下げを急がず", "利上げ見送り".
JA_REVERSE = re.compile(r"(?:観測|期待|予想|見通し|機運|織り込み|姿勢)?(?:の|が|は|を|も)?\s?"
                        r"(?:後退|遠の|剥落|はく落|見送|急が|慎重|否定|打ち消|織り込み過ぎ|せず|しない|休止|停止|終了|縮小|解除|出口|修正)")
JA_YIELD_UP = r"米(?:国)?の?(?:長期)?金利(?:が|は)?[^、。]{0,6}(?:上昇|高水準|最高)|米国債利回り(?:が|は)?[^、。]{0,4}上昇"
JA_YIELD_DOWN = r"米(?:国)?の?(?:長期)?金利(?:が|は)?[^、。]{0,6}(?:低下|下落)|米国債利回り(?:が|は)?[^、。]{0,4}低下"
JA_RISK_OFF = r"地政学|紛争|戦争|攻撃|ミサイル|暴落|リスクオフ|有事|緊張|制裁(?!金)|関税"
JA_RISK_ON = r"株高|最高値|リスクオン|停戦|合意"
JA_INTERVENE = r"為替介入|介入|レートチェック"
JA_VERBAL = r"けん制|牽制|過度な変動|投機的な動き|憂慮"
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
                if not JA_NOT_MOVE.match(t, i + len(word)):      # "円高につながらず" is not a yen rise
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
        for m in JA_YEN_RATE.finditer(single):
            add("JPY", 1 if m.group("up") else -1, "fx_move")
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
        risk_off, risk_on, intervene, verbal = JA_RISK_OFF, JA_RISK_ON, JA_INTERVENE, JA_VERBAL
        market_ctx = JA_MARKET_CTX

        def reversed_at(m):
            return bool(JA_REVERSE.match(t, m.end()))
    else:
        t = title.lower()
        pairs = list(EN_PAIR.finditer(t))
        # Tokens inside "USD/JPY" belong to the pair, not to one currency.
        ents = [e for e in _entities_en(t) if not any(p.start() <= e[0] < p.end() for p in pairs)]
        for s, e, cur in ents:
            if cur in CURRENCIES and not EN_CUR_WORD.fullmatch(t[s:e]):
                continue                                    # "Fed", "Australian", "US": not the currency itself
            d = 0
            m = EN_VERB.search(t, e, e + 40)
            if m and EN_CLAUSE_BREAK.search(t, e, m.start()):
                m = None                                    # "yen range; falling yields": another clause's verb
            if m and m.start() - e <= 25 and not any(e <= s2 < m.start() for s2, _, _ in ents):
                d = 1 if m.group("up") else -1
                if EN_NEG_MOVE.search(t[e:m.start()]):
                    d = 0                                   # "yen fails to rally", "dollar not falling"
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
                continue                    # "rupee falls against the dollar" says little about the majors
            add(cur, d, "fx_move")
            if (a := EN_AGAINST.search(t, e, e + 70)) and EN_AGAINST_CUR[a.group(1)] != cur:
                add(EN_AGAINST_CUR[a.group(1)], -d, "fx_move")      # "yen jumps against the dollar"
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
        risk_off, risk_on, intervene, verbal = EN_RISK_OFF, EN_RISK_ON, EN_INTERVENE, EN_VERBAL
        market_ctx = EN_MARKET_CTX

        def reversed_at(m):
            before = t[max(0, m.start() - 40):m.start()]
            return bool(EN_REVERSE_BEFORE.search(before) or EN_REVERSE_AFTER.match(t, m.end())
                        or EN_NEG_CUE.search(t[max(0, m.start() - 30):m.start()]) or EN_REVERSE_AFTER_FADE.match(t, m.end())
                        or EN_REVERSE_BEFORE_DEFLATE.search(before))

    for pattern, value, topic in cues:
        for m in re.finditer(pattern, t):
            cur = _nearest(ents, m.start()) or default_cur
            if topic == "policy" and reversed_at(m):
                value_m = -value
            else:
                value_m = value
            add(cur, value_m, topic)
    if re.search(yield_up, t):
        add("USD", 0.6, "yields")
    if re.search(yield_down, t):
        add("USD", -0.6, "yields")
    if market_ctx.search(t):
        on = re.search(risk_on, t)
        undone = bool(on) and lang != "ja" and bool(EN_RISK_ON_UNDONE.match(t, on.end()))
        if re.search(risk_off, t) or undone:
            for cur, v in RISK_OFF_EFFECT.items():
                add(cur, v, "risk")
        if on and not undone:
            for cur, v in RISK_ON_EFFECT.items():
                add(cur, v, "risk")
    mentioned = sorted({c for _, _, c in ents if c in CURRENCIES} | ({default_cur} if default_cur else set()))
    # An intervention mention supports the yen unless the headline already says how the yen moved.
    if re.search(intervene, t) and "JPY" in mentioned and "fx_move" not in topics:
        add("JPY", 0.8, "intervention")
    elif re.search(verbal, t) and "JPY" in mentioned and "fx_move" not in topics:
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
