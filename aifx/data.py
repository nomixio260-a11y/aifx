"""Daily FX price data: download, fallback and on-disk cache.

Primary source is Yahoo Finance's chart endpoint (daily OHLC). When it is
unavailable (rate limits are common) the ECB reference rates published by
the Frankfurter API are used instead; those only carry one rate per day, so
open/high/low are set equal to the close.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

USER_AGENT = "Mozilla/5.0 (compatible; aifx/0.1; +https://github.com/nomixio260-a11y/aifx)"
OHLC = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class Pair:
    code: str      # "USDJPY"
    base: str      # "USD"
    quote: str     # "JPY"
    name: str      # display name, e.g. "米ドル/円"

    @property
    def label(self) -> str:
        return f"{self.base}/{self.quote}"

    @property
    def yahoo_symbol(self) -> str:
        return f"{self.code}=X"

    @property
    def decimals(self) -> int:
        return 3 if self.quote == "JPY" else 5

    @property
    def pip(self) -> float:
        return 0.01 if self.quote == "JPY" else 0.0001


PAIRS: dict[str, Pair] = {
    p.code: p
    for p in [
        Pair("USDJPY", "USD", "JPY", "米ドル/円"),
        Pair("EURJPY", "EUR", "JPY", "ユーロ/円"),
        Pair("GBPJPY", "GBP", "JPY", "ポンド/円"),
        Pair("AUDJPY", "AUD", "JPY", "豪ドル/円"),
        Pair("EURUSD", "EUR", "USD", "ユーロ/米ドル"),
        Pair("GBPUSD", "GBP", "USD", "ポンド/米ドル"),
        Pair("AUDUSD", "AUD", "USD", "豪ドル/米ドル"),
    ]
}


def get_pair(code: str) -> Pair:
    key = code.upper().replace("/", "").replace("_", "")
    if key not in PAIRS:
        raise KeyError(f"unknown pair {code!r}; choose from {', '.join(PAIRS)}")
    return PAIRS[key]


def _http_json(url: str, timeout: float = 20.0, retries: int = 3) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network, HTTP 429, bad JSON
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last}")


def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    """Turn a Yahoo v8 chart response into a daily OHLC frame."""
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    tz = result.get("meta", {}).get("exchangeTimezoneName") or "Europe/London"
    idx = pd.to_datetime(result["timestamp"], unit="s", utc=True)
    df = pd.DataFrame({k: quote.get(k) for k in OHLC}, index=idx, dtype="float64")
    # Daily FX bars are stamped at local midnight of the exchange (23:00 UTC in
    # British summer time), so take the calendar date in that timezone.
    df.index = df.index.tz_convert(tz).tz_localize(None).normalize()
    df.index.name = "date"
    df = df.dropna(subset=["close"])
    return _clean(df)


def fetch_yahoo(pair: Pair, years: int = 5) -> pd.DataFrame:
    url = (
        "https://query2.finance.yahoo.com/v8/finance/chart/"
        + urllib.parse.quote(pair.yahoo_symbol)
        + f"?range={years}y&interval=1d"
    )
    return parse_yahoo_chart(_http_json(url))


def parse_frankfurter(payload: dict, quote: str) -> pd.DataFrame:
    rates = payload["rates"]
    idx = pd.to_datetime(sorted(rates))
    close = np.array([rates[d.strftime("%Y-%m-%d")][quote] for d in idx], dtype="float64")
    df = pd.DataFrame({k: close for k in OHLC}, index=idx)
    df.index.name = "date"
    return _clean(df)


def fetch_frankfurter(pair: Pair, years: int = 5) -> pd.DataFrame:
    start = (date.today() - timedelta(days=365 * years + 7)).isoformat()
    url = f"https://api.frankfurter.app/{start}..?from={pair.base}&to={pair.quote}"
    return parse_frankfurter(_http_json(url), pair.quote)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index.dayofweek < 5]  # FX closes at weekends; drop stray Sunday bars
    # Fill partial bars and keep high/low consistent with open/close.
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["high"] = df[OHLC].max(axis=1)
    df["low"] = df[OHLC].min(axis=1)
    return df[OHLC].astype("float64")


def cache_path(cache_dir: Path, pair: Pair) -> Path:
    return Path(cache_dir) / f"{pair.code}.csv"


def load_cached(cache_dir: Path, pair: Pair) -> pd.DataFrame | None:
    path = cache_path(cache_dir, pair)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return _clean(df)


def save_cache(cache_dir: Path, pair: Pair, df: pd.DataFrame) -> None:
    path = cache_path(cache_dir, pair)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="date", float_format="%.6f")


def load_prices(
    pair: Pair,
    cache_dir: Path | str = "data/cache",
    offline: bool = False,
    years: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Return (ohlc frame, source name). Falls back to the cache when offline or on failure."""
    cache_dir = Path(cache_dir)
    if not offline:
        errors = []
        for source, fetch in (("Yahoo Finance", fetch_yahoo), ("ECB (Frankfurter)", fetch_frankfurter)):
            try:
                df = fetch(pair, years=years)
                if len(df) < 300:
                    raise RuntimeError(f"only {len(df)} rows")
                save_cache(cache_dir, pair, df)
                return df, source
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        cached = load_cached(cache_dir, pair)
        if cached is not None:
            return cached, "cache"
        raise RuntimeError(f"{pair.code}: no data. " + " | ".join(errors))
    cached = load_cached(cache_dir, pair)
    if cached is None:
        raise RuntimeError(f"{pair.code}: no cached data in {cache_dir}; run without --offline first")
    return cached, "cache"


def synthetic_prices(
    n: int = 800,
    start_price: float = 150.0,
    drift: float = 0.0,
    vol: float = 0.006,
    seed: int = 0,
    end: str | None = None,
) -> pd.DataFrame:
    """Random-walk OHLC for demos and tests (no network needed)."""
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.standard_normal((2, n))) * vol * 0.6
    high = np.maximum(open_, close) * (1 + wiggle[0])
    low = np.minimum(open_, close) * (1 - wiggle[1])
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
