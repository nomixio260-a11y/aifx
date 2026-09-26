from datetime import datetime, timedelta, timezone

import pytest

from aifx import news
from aifx.timeutil import iso

UTC = timezone.utc
NOW = datetime(2026, 9, 24, 6, 0, tzinfo=UTC)

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Dollar jumps as Fed turns hawkish - Reuters</title><link>https://x/1</link>
<pubDate>Thu, 24 Sep 2026 05:00:00 GMT</pubDate><source url="https://reuters.com">Reuters</source></item>
</channel></rss>"""
ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>ECB holds rates</title><link href="https://x/2"/><updated>2026-09-24T04:00:00Z</updated></entry></feed>"""
RDF = b"""<?xml version="1.0"?><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns="http://purl.org/rss/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/">
<item><title>RBA media release</title><link>https://x/3</link><dc:date>2026-09-24T11:30:00+10:00</dc:date></item></rdf:RDF>"""


def test_parses_rss_atom_and_rdf():
    rss = news.parse_feed(RSS)
    assert rss[0]["publisher"] == "Reuters" and rss[0]["published"] == datetime(2026, 9, 24, 5, tzinfo=UTC)
    assert news.parse_feed(ATOM)[0]["link"] == "https://x/2"
    assert news.parse_feed(RDF)[0]["published"] == datetime(2026, 9, 24, 1, 30, tzinfo=UTC)
    item = news.normalise(rss, news.SOURCES[4], NOW)[0]
    assert item["title"] == "Dollar jumps as Fed turns hawkish"  # publisher suffix removed
    assert item["an"]["cur"]["USD"] > 0


@pytest.mark.parametrize("title,lang,expect", [
    ("USD/JPY outlook: Hawkish Fed recalibration pressures the yen", "en", {"USD": 1, "JPY": -1}),
    ("GBP/USD forecast: US dollar surges as bonds implode", "en", {"USD": 1}),
    ("Rupee falls against the US dollar", "en", {}),          # emerging-market moves say little about the majors
    ("USD/JPY rises above 158", "en", {"USD": 1, "JPY": -1}),
    ("ECB signals rate cuts ahead; euro slides", "en", {"EUR": -1}),
    ("Missile attack sparks safe-haven demand", "en", {"JPY": 1, "AUD": -1}),
    ("円急落、米長期金利が上昇", "ja", {"JPY": -1, "USD": 1}),
    ("米国の長期金利が19年ぶりの高水準に 円は下落", "ja", {"JPY": -1, "USD": 1}),
    ("円安修正が本格化", "ja", {"JPY": 1}),
    ("日銀、利上げ検討へ", "ja", {"JPY": 1}),
    ("ドル円は反落、米雇用統計が予想を下回る", "ja", {"USD": -1, "JPY": 1}),
    ("Googleに4億ユーロのEU制裁金", "ja", {}),
    ("九州で豪雨 単独事故も", "ja", {}),
    # policy expectations turned around
    ("利上げ観測が後退、ドル売り優勢", "ja", {"USD": -1}),
    ("利下げ観測後退でドル買い", "ja", {"USD": 1}),
    ("FRB、利下げを急がず", "ja", {"USD": 1}),
    ("日銀、利上げ見送り", "ja", {"JPY": -1}),
    ("ECB、利下げ見送り", "ja", {"EUR": 1}),
    ("日銀、金融緩和を縮小", "ja", {"JPY": 1}),
    ("Fed signals no rush to cut rates", "en", {"USD": 1}),
    ("Traders pare bets on Fed rate cuts after strong jobs data", "en", {"USD": 1}),
    ("Rate cut bets fade as US inflation stays hot", "en", {"USD": 1}),
    ("BOJ unlikely to hike this year", "en", {"JPY": -1}),
    ("ECB rules out further rate hikes", "en", {"EUR": -1}),
    ("Fed ends rate hikes", "en", {"USD": -1}),
    # flows, data, yen rate and verbal intervention
    ("ドル買い優勢、ドル円は158円に迫る 原油安は円高につながらず", "ja", {"USD": 1}),
    ("米雇用統計、予想を大幅に上回る", "ja", {"USD": 1}),
    ("円相場、一時158円台前半まで下落 米利上げ観測でレートチェック効果を相殺", "ja", {"JPY": -1, "USD": 1}),
    ("片山財務相、為替の過度な変動をけん制", "ja", {"JPY": 1}),
    ("Japan warns against rapid yen moves", "en", {"JPY": 1}),
])
def test_keyword_analysis_directions(title, lang, expect):
    got = news.analyze_lexicon(title, lang)["cur"]
    if "円高につながらず" in title:
        assert "JPY" not in got
    assert set(got) >= set(expect)
    for cur, sign in expect.items():
        assert got[cur] * sign > 0, (cur, got)
    if not expect:
        assert got == {}


def item(title, published, fetched, scores, src="gn-en-fx"):
    return {"id": title, "src": src, "title": title, "published_at": iso(published), "fetched_at": iso(fetched),
            "an": {"by": "t", "cur": scores, "men": list(scores), "top": []}}


def test_only_headlines_stored_before_the_cutoff_count():
    items = [
        item("old but stored early", NOW - timedelta(hours=2), NOW - timedelta(hours=1), {"USD": 1.0}),
        item("stored at the cutoff", NOW - timedelta(hours=1), NOW, {"USD": -1.0}),
        item("published after cutoff", NOW + timedelta(minutes=5), NOW - timedelta(minutes=1), {"USD": -1.0}),
        item("too old", NOW - timedelta(hours=60), NOW - timedelta(hours=59), {"USD": -1.0}),
    ]
    got = news.eligible(items, NOW)
    assert [it["title"] for it in got] == ["old but stored early"]
    press = news.pressures(items, NOW)
    assert press["USD"]["p"] > 0 and press["USD"]["n"] == 1
    assert news.pair_signal(press, "USD", "JPY") == press["USD"]["p"]
    assert news.pair_signal(press, "EUR", "USD") == -press["USD"]["p"]


def test_pressure_is_shrunk_when_there_are_few_headlines():
    one = news.pressures([item("a", NOW - timedelta(hours=1), NOW - timedelta(hours=1), {"JPY": 1.0})], NOW)
    many = news.pressures([item(str(i), NOW - timedelta(hours=1), NOW - timedelta(hours=1), {"JPY": 1.0})
                           for i in range(20)], NOW)
    assert 0 < one["JPY"]["p"] < many["JPY"]["p"] < 1


def test_calendar_keeps_tracked_high_and_medium_events():
    payload = [
        {"title": "CPI", "country": "USD", "date": "2026-09-24T08:30:00-04:00", "impact": "High", "forecast": "", "previous": ""},
        {"title": "Retail", "country": "CAD", "date": "2026-09-24T08:30:00-04:00", "impact": "High"},
        {"title": "Holiday", "country": "JPY", "date": "2026-09-24T00:00:00-04:00", "impact": "Holiday"},
    ]
    evs = news.parse_calendar(payload)
    assert [e["title"] for e in evs] == ["CPI"] and evs[0]["time"] == "2026-09-24T12:30:00Z"


def test_syndicated_copies_count_as_one_story():
    base = NOW - timedelta(hours=2)
    copies = [item(t, base + timedelta(minutes=i), base + timedelta(minutes=i + 1), {"USD": 1.0})
              for i, t in enumerate(["東京為替：ドル・円は底堅い、円買いは一服 執筆： Fisco",
                                     "東京為替：ドル・円は底堅い、円買いは一服",
                                     "東京為替：ドル・円は底堅い、円買いは一服(フィスコ)"])]
    other = item("Fed signals no rush to cut rates", base, base, {"USD": 1.0})
    st = news.stories(copies + [other])
    assert len({st[c["id"]] for c in copies}) == 1 and st[other["id"]] != st[copies[0]["id"]]
    three = news.pressures(copies + [other], NOW)["USD"]
    one = news.pressures(copies[:1] + [other], NOW)["USD"]
    assert three["n"] == one["n"] == 2
    assert three["w"] == pytest.approx(one["w"], rel=0.05)


def _sign(title, lang="en"):
    return {c: (1 if v > 0 else -1) for c, v in news.analyze_lexicon(title, lang)["cur"].items()}


@pytest.mark.parametrize("title, want", [
    # risk words count only in market headlines (war and politics stories are not market news)
    ("Hegseth estimates Iran war has cost $48.5b so far", {}),
    ("Mortgage Rates Rise as Iran Ceasefire Crumbles", {}),
    ("Dow drops 500 points as oil nears $100 amid Iran war", {"JPY": 1, "AUD": -1, "USD": 1}),
    # an undone ceasefire is risk-off
    ("Oil prices rise 7%, and Dow drops 600 points after Trump says ceasefire with Iran is 'over'", {"JPY": 1, "AUD": -1, "USD": 1}),
    # emerging-market currencies say nothing about the dollar against the majors
    ("Rupee falls 13 paise to 95.56 against U.S. dollar in early trade", {}),
    # negation and fading expectations turn a policy cue around
    ("Fed Didn't Raise Rates After All — Will Mortgage Rates Fall? | National", {"USD": -1}),
    ("Asian markets choppy as US jobs data douse Fed rate hike bets", {"USD": -1}),
    # a move verb moves only a currency named as a currency
    ("Cotality Report: Australian property resale gains hit record $377,000", {}),
    ("World stocks are mixed as yen jumps against the dollar, while oil prices slip", {"JPY": 1, "USD": -1}),
    # names in office in 2026, and quieter move verbs
    ("U.S. Dollar Moves Lower As Bessent Boosts Bond Buybacks", {"USD": -1}),
    ("Why Treasury yields are ripping higher", {"USD": 1}),
])
def test_headline_reading_fixes(title, want):
    assert _sign(title) == want


def test_a_move_verb_in_another_clause_does_not_move_the_currency():
    t = "[Tokyo Forex] Dollar trades in the lower 158 yen range; falling U.S. long-term yields also exert downward pressure"
    assert "JPY" not in news.analyze_lexicon(t, "en")["cur"]
