"""Typed views over the append-only data files: price bars, news items, calendar events.

Everything the forecaster uses is read back from these files, never from the
raw download, so a verifier holding the same files sees exactly the inputs
the forecaster saw.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta

import pandas as pd

from .ledger import DataFiles, canonical, sha256_hex
from .timeutil import UTC, iso, parse_iso

PRICE_COLS = ["open", "high", "low", "close"]
INTRADAY_TFS = ("15m", "1h")     # bars keyed by their UTC open time; others by date


def parse_price_lines(lines: list[str], tf: str) -> pd.DataFrame:
    """Parse a price file's lines (header first) into an OHLC frame."""
    if len(lines) < 2:
        return pd.DataFrame(columns=PRICE_COLS, dtype="float64")
    df = pd.read_csv(io.StringIO("\n".join(lines)), index_col=0)
    if tf in INTRADAY_TFS:
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "time"
    else:
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
    return df.astype("float64")


class PriceStore:
    def __init__(self, files: DataFiles):
        self.files = files
        self._cache: dict[str, pd.DataFrame] = {}

    @staticmethod
    def path(pair: str, tf: str) -> str:
        return f"prices/{pair}_{tf}.csv"

    def load(self, pair: str, tf: str) -> pd.DataFrame:
        path = self.path(pair, tf)
        if path not in self._cache:
            self._cache[path] = parse_price_lines(self.files.read_lines(path), tf)
        return self._cache[path]

    def append_new(self, pair: str, tf: str, df: pd.DataFrame) -> int:
        """Append bars newer than the last stored one. Returns how many were added."""
        path = self.path(pair, tf)
        have = self.load(pair, tf)
        if len(have):
            df = df[df.index > have.index[-1]]
        if not len(df):
            return 0
        intraday = tf in INTRADAY_TFS
        lines = [] if self.files.lines(path) else [("time" if intraday else "date") + ",open,high,low,close"]
        for ts, row in df.iterrows():
            key = iso(ts.to_pydatetime()) if intraday else ts.strftime("%Y-%m-%d")
            lines.append(key + "," + ",".join(f"{float(row[c]):.6f}" for c in PRICE_COLS))
        self.files.append_lines(path, lines)
        self._cache.pop(path, None)
        return len(df)


def _month_paths(files: DataFiles, folder: str) -> list[str]:
    d = files.root / folder
    if not d.exists():
        return []
    return [f"{folder}/{p.name}" for p in sorted(d.glob("*.jsonl"))]


class JsonlStore:
    """Monthly JSONL files of dict items with a stable ``id`` for de-duplication."""

    folder = ""

    def __init__(self, files: DataFiles):
        self.files = files

    def load(self, since: datetime | None = None) -> list[dict]:
        out = []
        for path in _month_paths(self.files, self.folder):
            if since is not None and path.split("/")[-1][:7] < iso(since)[:7]:
                continue
            for line in self.files.read_lines(path):
                if line.strip():
                    out.append(json.loads(line))
        return out

    def append(self, items: list[dict], at: datetime) -> list[dict]:
        """Append items not seen in the last ~2 months; stamps ``fetched_at``."""
        seen = {it["id"] for it in self.load(since=at - timedelta(days=62))}
        fresh = []
        for it in items:
            if it["id"] in seen:
                continue
            seen.add(it["id"])
            item = dict(it)
            item["fetched_at"] = iso(at)
            fresh.append(item)
        if fresh:
            path = f"{self.folder}/{iso(at)[:7]}.jsonl"
            self.files.append_lines(path, [canonical(it).decode("utf-8") for it in fresh])
        return fresh


class NewsStore(JsonlStore):
    folder = "news"


class CalendarStore(JsonlStore):
    folder = "calendar"


def stable_id(*parts: str) -> str:
    return sha256_hex("\x1f".join(parts).encode("utf-8"))[:16]


def known_before(items: list[dict], cutoff: datetime) -> list[dict]:
    """Items the system had already stored strictly before ``cutoff``."""
    c = iso(cutoff)
    return [it for it in items if it["fetched_at"] < c]


def parse_time(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return parse_iso(s)
    except ValueError:
        return None


__all__ = ["PriceStore", "NewsStore", "CalendarStore", "stable_id", "known_before", "parse_time", "UTC"]
