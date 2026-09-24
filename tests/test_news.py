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
    ("Rupee falls against the US dollar", "en", {"USD": 1}),
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
])
def test_keyword_analysis_directions(title, lang, expect):
    got = news.analyze_lexicon(title, lang)["cur"]
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


class _FakeBlock:
    def __init__(self, text):
        self.type, self.text = "text", text


class _FakeResp:
    def __init__(self, text, stop="end_turn"):
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 5})()


class _FakeClient:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kw):
                outer.calls.append(kw)
                return outer.resp

        self.beta = type("B", (), {"messages": _Messages()})()


def test_claude_analyzer_uses_structured_output_and_keeps_keyword_scores():
    pytest.importorskip("anthropic")
    import json

    from aifx.news_llm import ClaudeAnalyzer, merge
    it = news.normalise(news.parse_feed(RSS), news.SOURCES[4], NOW)[0]
    body = json.dumps({"items": [{"id": it["id"], "impacts": [{"currency": "USD", "direction": 0.8, "confidence": 0.5}],
                                  "category": "policy", "summary_ja": "FRBのタカ派姿勢でドル高"}]})
    client = _FakeClient(_FakeResp(body))
    an = ClaudeAnalyzer(model="claude-opus-5", client=client)
    out, err = an.analyze([it])
    assert err is None and out[it["id"]]["cur"] == {"USD": 0.4}
    call = client.calls[0]
    assert call["model"] == "claude-opus-5" and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    merged = merge(it, out[it["id"]])
    assert merged["an"]["by"] == "claude:claude-opus-5" and merged["an"]["lex"] == it["an"]["cur"]


def test_claude_refusal_falls_back_to_keywords():
    pytest.importorskip("anthropic")
    from aifx.news_llm import ClaudeAnalyzer
    it = news.normalise(news.parse_feed(RSS), news.SOURCES[4], NOW)[0]
    an = ClaudeAnalyzer(client=_FakeClient(_FakeResp("", stop="refusal")))
    out, err = an.analyze([it])
    assert out == {} and "declined" in err
