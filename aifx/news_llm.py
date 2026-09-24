"""Optional headline analysis with Claude.

Enabled when ``ANTHROPIC_API_KEY`` (or ``ANTHROPIC_AUTH_TOKEN``) is set, unless
``AIFX_LLM=off``. Each new relevant headline is analysed once; the result is
stored with the headline, so forecasts stay reproducible without calling the
API again. If the call fails or is declined, the keyword analysis is kept.

Claude only sees headline text published before the forecast origin and is
used for live forecasts only. Its weight in the forecast is learned from live
results, never from a backtest: a language model may already know how
historical markets moved, which would make a backtest of it meaningless.
"""

from __future__ import annotations

import json
import os

DEFAULT_MODEL = "claude-opus-5"
MAX_ITEMS = 40

SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "impacts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "currency": {"type": "string", "enum": ["USD", "JPY", "EUR", "GBP", "AUD"]},
                                "direction": {"type": "number"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["currency", "direction", "confidence"],
                            "additionalProperties": False,
                        },
                    },
                    "category": {"type": "string", "enum": [
                        "policy", "data", "yields", "risk", "intervention", "politics", "fx_move", "other"]},
                    "summary_ja": {"type": "string"},
                },
                "required": ["id", "impacts", "category", "summary_ja"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["items"],
    "additionalProperties": False,
}

SYSTEM = """You assess financial news headlines for their likely effect on five currencies over the next hours to days: USD, JPY, EUR, GBP, AUD.

For each headline, return the currencies it plausibly affects. direction is from -1 (the currency weakens) to +1 (it strengthens), for the currency itself, not a pair: "USD/JPY rises" means USD +, JPY -. confidence is 0 to 1. Leave impacts empty when a headline has no clear currency implication, such as routine administrative notices.

Judge only from the headline text. Do not use any knowledge of how markets moved after the headline was published.

Useful regularities: hawkish central-bank news, strong data and rising yields support that country's currency; dovish news, weak data and political instability weigh on it. Risk-off shocks (war, crisis, market crashes) tend to support JPY and to a lesser degree USD, and weigh on AUD. Signals of Japanese currency intervention support JPY.

summary_ja: one short Japanese sentence stating what the headline means for the currencies."""


def enabled() -> bool:
    if os.environ.get("AIFX_LLM", "").lower() == "off":
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("AIFX_LLM", "").lower() == "on")


class ClaudeAnalyzer:
    def __init__(self, model: str | None = None, max_items: int | None = None, client=None):
        import anthropic

        self.anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model or os.environ.get("AIFX_LLM_MODEL", DEFAULT_MODEL)
        self.max_items = max_items or int(os.environ.get("AIFX_LLM_MAX_ITEMS", MAX_ITEMS))
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    @property
    def name(self) -> str:
        return f"claude:{self.model}"

    def analyze(self, items: list[dict]) -> tuple[dict[str, dict], str | None]:
        """Returns ({item id: analysis}, error message or None)."""
        batch = items[: self.max_items]
        if not batch:
            return {}, None
        payload = [{"id": it["id"], "title": it["title"], "publisher": it["publisher"],
                    "published_at": it["published_at"]} for it in batch]
        anthropic = self.anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except anthropic.RateLimitError:
            return {}, "Claude: rate limited"
        except anthropic.APIStatusError as exc:
            return {}, f"Claude: API error {exc.status_code}"
        except anthropic.APIConnectionError:
            return {}, "Claude: connection error"
        self.usage["calls"] += 1
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        if resp.stop_reason == "refusal":
            return {}, "Claude: request declined"
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}, "Claude: unreadable response"
        wanted = {it["id"] for it in batch}
        out = {}
        for row in data.get("items", []):
            if row.get("id") not in wanted:
                continue
            cur = {}
            for imp in row.get("impacts", []):
                d = max(-1.0, min(1.0, float(imp["direction"])))
                c = max(0.0, min(1.0, float(imp["confidence"])))
                if abs(d * c) > 1e-6:
                    cur[imp["currency"]] = round(cur.get(imp["currency"], 0.0) + d * c, 3)
            out[row["id"]] = {
                "by": self.name,
                "cur": {k: max(-1.0, min(1.0, v)) for k, v in sorted(cur.items())},
                "top": [row.get("category", "other")],
                "ja": row.get("summary_ja", "")[:120],
            }
        return out, None


def merge(item: dict, llm: dict) -> dict:
    """Keep the keyword analysis for comparison and use Claude's scores."""
    an = dict(llm)
    an["men"] = sorted(set(item["an"].get("men", [])) | set(llm["cur"]))
    an["lex"] = item["an"]["cur"]
    out = dict(item)
    out["an"] = an
    return out
