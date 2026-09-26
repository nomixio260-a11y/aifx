"""FX price data: download, completed-bar filtering, fallbacks and a synthetic market.

Primary source is Yahoo Finance's chart endpoint. Hourly bars drive the live
forecasts and the scoring of every prediction; daily bars (5 years) train the
daily models. When Yahoo is unavailable for daily data the ECB reference rates
from the Frankfurter API are used instead (one rate per day, so open/high/low
equal the close).

Only *completed* bars are ever returned to the forecasting code: a bar that is
still forming at fetch time would leak information that did not exist at the
forecast's origin.
"""

from __future__ import annotations

import math

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .timeutil import UTC, is_market_open, london_day_end

USER_AGENT = "Mozilla/5.0 (compatible; aifx/0.2; +https://github.com/nomixio260-a11y/aifx)"
OHLC = ["open", "high", "low", "close"]
YAHOO_HOSTS = ("query2.finance.yahoo.com", "query1.finance.yahoo.com")
# A bar only counts as complete a little after its nominal end, so late ticks settle.
SETTLE = timedelta(minutes=3)


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
CURRENCIES = ("USD", "JPY", "EUR", "GBP", "AUD")


def get_pair(code: str) -> Pair:
    key = code.upper().replace("/", "").replace("_", "")
    if key not in PAIRS:
        raise KeyError(f"unknown pair {code!r}; choose from {', '.join(PAIRS)}")
    return PAIRS[key]


def http_get(url: str, timeout: float = 20.0, retries: int = 3, accept: str = "*/*") -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # network, HTTP 4xx/5xx
            last = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}: {last}")


def _http_json(url: str, timeout: float = 20.0, retries: int = 3) -> dict:
    return json.loads(http_get(url, timeout=timeout, retries=retries, accept="application/json").decode("utf-8"))


def _yahoo_chart(symbol: str, query: str) -> dict:
    errors = []
    for host in YAHOO_HOSTS:
        url = f"https://{host}/v8/finance/chart/{urllib.parse.quote(symbol)}?{query}"
        try:
            return _http_json(url, retries=2)
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors))


# ---------------------------------------------------------------- daily bars

def parse_yahoo_chart(payload: dict) -> pd.DataFrame:
    """Turn a Yahoo v8 daily chart response into a daily OHLC frame (index: London date)."""
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
    return _clean_daily(df)


def fetch_yahoo(pair: Pair, years: int = 5) -> pd.DataFrame:
    return parse_yahoo_chart(_yahoo_chart(pair.yahoo_symbol, f"range={years}y&interval=1d"))


def parse_frankfurter(payload: dict, quote: str) -> pd.DataFrame:
    rates = payload["rates"]
    idx = pd.to_datetime(sorted(rates))
    close = np.array([rates[d.strftime("%Y-%m-%d")][quote] for d in idx], dtype="float64")
    df = pd.DataFrame({k: close for k in OHLC}, index=idx)
    df.index.name = "date"
    return _clean_daily(df)


def fetch_frankfurter(pair: Pair, years: int = 5) -> pd.DataFrame:
    start = (date.today() - timedelta(days=365 * years + 7)).isoformat()
    url = f"https://api.frankfurter.app/{start}..?from={pair.base}&to={pair.quote}"
    return parse_frankfurter(_http_json(url), pair.quote)


def _clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index.dayofweek < 5]  # FX closes at weekends; drop stray Sunday bars
    return _fix_ohlc(df)


def _fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["high"] = df[OHLC].max(axis=1)
    df["low"] = df[OHLC].min(axis=1)
    return df[OHLC].astype("float64")


def london_days(daily: pd.DataFrame, hourly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily bars that end when the forecasts' London day ends.

    Yahoo's daily FX bars have, since about 2011, a close equal to the price at
    00:00 UTC at the start of the labelled day (open and close are nearly the
    same; the high and low do cover the day). Each bar is rebuilt here from the
    hourly bars of its London business day where they cover it (the Sunday
    evening counts towards Monday), and elsewhere its close is taken from the
    next bar's open (the price at the start of the next day). Only prices that
    were known when the bar ended are used, so the same bars come back from the
    committed data at any later time.
    """
    if not len(daily):
        return daily
    o, h, lo, c = (daily[k].to_numpy(dtype=float).copy() for k in OHLC)
    c[:-1] = o[1:]                                        # the next day's opening price
    if hourly is not None and len(hourly):
        ends = pd.DatetimeIndex([london_day_end(d.date()) for d in daily.index])
        h_end = (hourly.index + pd.Timedelta(hours=1)).as_unit("ns")
        ho, hh, hl, hc = (hourly[k].to_numpy(dtype=float) for k in OHLC)
        prev_end = ends[0] - pd.Timedelta(days=1)
        for i, end in enumerate(ends):
            a = int(h_end.searchsorted(prev_end, side="right"))
            b = int(h_end.searchsorted(end, side="right"))
            if b - a >= 12 and end - h_end[b - 1] <= pd.Timedelta(hours=3):
                o[i], h[i], lo[i], c[i] = ho[a], hh[a:b].max(), hl[a:b].min(), hc[b - 1]
            prev_end = end
    out = pd.DataFrame({"open": o, "high": np.maximum.reduce([h, o, c]), "low": np.minimum.reduce([lo, o, c]),
                        "close": c}, index=daily.index)
    return out


def completed_daily(df: pd.DataFrame, cutoff: datetime) -> pd.DataFrame:
    """Daily bars whose London day has ended (plus settling time) by ``cutoff``."""
    keep = [london_day_end(d.date()) + SETTLE <= cutoff for d in df.index]
    return df[np.array(keep, dtype=bool)]


def fetch_daily(pair: Pair, years: int = 5) -> tuple[pd.DataFrame, str]:
    errors = []
    for source, fetch in (("Yahoo Finance", fetch_yahoo), ("ECB (Frankfurter)", fetch_frankfurter)):
        try:
            df = fetch(pair, years=years)
            if len(df) < 300:
                raise RuntimeError(f"only {len(df)} rows")
            return df, source
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    raise RuntimeError(f"{pair.code}: no daily data. " + " | ".join(errors))


# --------------------------------------------------------------- hourly bars

def parse_yahoo_intraday(payload: dict, cutoff: datetime, minutes: int = 60) -> tuple[pd.DataFrame, dict]:
    """Completed intraday bars (index: bar open time, UTC) and the live quote.

    Yahoo appends the live, still-forming bar and sometimes an extra point
    stamped with the quote time; both are dropped here.
    """
    result = payload["chart"]["result"][0]
    meta = result.get("meta", {})
    quote = result["indicators"]["quote"][0]
    ts = result.get("timestamp") or []
    idx = pd.to_datetime(ts, unit="s", utc=True)
    df = pd.DataFrame({k: quote.get(k) for k in OHLC}, index=idx, dtype="float64")
    df.index.name = "time"
    step = pd.Timedelta(minutes=minutes)
    aligned = (df.index.minute % minutes == 0) & (df.index.second == 0) if minutes < 60 else (
        (df.index.minute == 0) & (df.index.second == 0))
    done = (df.index + step + SETTLE) <= pd.Timestamp(cutoff)
    df = df[aligned & done].dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    live = {
        "price": meta.get("regularMarketPrice"),
        "time": meta.get("regularMarketTime"),
    }
    return _fix_ohlc(df), live


def fetch_intraday(pair: Pair, cutoff: datetime, minutes: int, range_: str) -> tuple[pd.DataFrame, dict]:
    interval = "60m" if minutes == 60 else f"{minutes}m"
    payload = _yahoo_chart(pair.yahoo_symbol, f"range={range_}&interval={interval}")
    return parse_yahoo_intraday(payload, cutoff, minutes)


def fetch_hourly(pair: Pair, cutoff: datetime, range_: str = "1y") -> tuple[pd.DataFrame, dict]:
    return fetch_intraday(pair, cutoff, 60, range_)


class YahooMarket:
    """Live market data source used by the server."""

    name = "Yahoo Finance"

    def hourly(self, pair: Pair, cutoff: datetime, range_: str = "1y"):
        return fetch_hourly(pair, cutoff, range_)

    def intraday(self, pair: Pair, cutoff: datetime, minutes: int, range_: str):
        return fetch_intraday(pair, cutoff, minutes, range_)

    def daily(self, pair: Pair, cutoff: datetime):
        df, source = fetch_daily(pair)
        return completed_daily(df, cutoff), source


# ------------------------------------------------------------------ offline

class SyntheticMarket:
    """Deterministic random-walk market for tests, demos and offline runs.

    Prices only depend on the seed and the bar time, so a bar never changes
    once it exists, exactly like a real completed bar.
    """

    name = "synthetic"

    def __init__(self, seed: int = 0, start: datetime | None = None, vol: float = 0.0012):
        self.seed = seed
        self.start = (start or datetime(2023, 1, 1, tzinfo=UTC))
        self.vol = vol
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}

    def _series(self, pair: Pair, end: datetime) -> pd.DataFrame:
        # Bars depend only on their position in time, so a longer series sliced
        # at ``end`` is identical to one generated up to ``end``.
        have = self._cache.get(pair.code)
        if have is None or have[0] < end:
            far = end + timedelta(days=30)
            have = (far, self._generate(pair, far))
            self._cache[pair.code] = have
        df = have[1]
        return df[df.index < pd.Timestamp(end)]

    def _generate(self, pair: Pair, end: datetime) -> pd.DataFrame:
        hours = pd.date_range(self.start, end, freq="h", tz=UTC, inclusive="left")
        hours = hours[[is_market_open(t.to_pydatetime()) for t in hours]]
        salt = sum(ord(c) * (i + 1) for i, c in enumerate(pair.code))
        n = len(hours)
        # One generator per column: the i-th draw never depends on how many bars exist.
        rets, up, down = (np.random.default_rng([self.seed, salt, k]).standard_normal(n) for k in range(3))
        rets = self.vol * rets
        base = 150.0 if pair.quote == "JPY" else 1.1
        close = base * np.exp(np.cumsum(rets))
        open_ = np.concatenate([[base], close[:-1]])
        df = pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) * (1 + np.abs(up) * self.vol * 0.5),
            "low": np.minimum(open_, close) * (1 - np.abs(down) * self.vol * 0.5), "close": close,
        }, index=hours)
        df.index.name = "time"
        return df

    def hourly(self, pair: Pair, cutoff: datetime, range_: str = "1y"):
        df = self._series(pair, cutoff)
        df = df[(df.index + pd.Timedelta(hours=1) + SETTLE) <= pd.Timestamp(cutoff)]
        live = {"price": float(df["close"].iloc[-1]) if len(df) else None, "time": int(cutoff.timestamp())}
        return df, live

    def intraday(self, pair: Pair, cutoff: datetime, minutes: int, range_: str = "1mo"):
        if minutes == 60:
            return self.hourly(pair, cutoff, range_)
        if 60 % minutes:
            raise ValueError(f"unsupported bar length {minutes}")
        h = self._series(pair, cutoff)
        q = self._subdivide(pair, h, 60 // minutes)
        q = q[(q.index + pd.Timedelta(minutes=minutes) + SETTLE) <= pd.Timestamp(cutoff)]
        if range_.endswith("d"):
            q = q[q.index >= pd.Timestamp(cutoff) - pd.Timedelta(days=int(range_[:-1]))]
        live = {"price": float(q["close"].iloc[-1]) if len(q) else None, "time": int(cutoff.timestamp())}
        return q, live

    def _subdivide(self, pair: Pair, h: pd.DataFrame, parts: int) -> pd.DataFrame:
        """Split each hourly bar into ``parts`` sub-bars along a Brownian bridge, so the
        sub-bars end exactly at the hourly closes (bars still depend only on their time)."""
        n = len(h)
        if not n:
            return h.copy()
        salt = sum(ord(c) * (i + 1) for i, c in enumerate(pair.code))
        z = np.random.default_rng([self.seed, salt, 11]).standard_normal((n, parts)) * self.vol / math.sqrt(parts)
        w = np.cumsum(z, axis=1)
        frac = np.arange(1, parts + 1) / parts
        bridge = w - frac[None, :] * w[:, -1:]
        lo, lc = np.log(h["open"].to_numpy()), np.log(h["close"].to_numpy())
        pts = lo[:, None] + (lc - lo)[:, None] * frac[None, :] + bridge
        close = np.exp(pts).ravel()
        open_ = np.exp(np.concatenate([lo[:, None], pts[:, :-1]], axis=1)).ravel()
        wig = np.abs(np.random.default_rng([self.seed, salt, 12]).standard_normal((n * parts, 2))) * self.vol * 0.25
        idx = (h.index.repeat(parts) + pd.to_timedelta(np.tile(np.arange(parts) * (60 // parts), n), unit="min"))
        df = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) * (1 + wig[:, 0]),
                           "low": np.minimum(open_, close) * (1 - wig[:, 1]), "close": close}, index=idx)
        df.index.name = "time"
        return df

    def daily(self, pair: Pair, cutoff: datetime):
        h = self._series(pair, cutoff)
        london = h.index.tz_convert("Europe/London")
        day = pd.DatetimeIndex(london.date)
        agg = h.groupby(day).agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        agg.index.name = "date"
        agg = agg[agg.index.dayofweek < 5]
        return completed_daily(agg, cutoff), "synthetic"


def synthetic_prices(
    n: int = 800,
    start_price: float = 150.0,
    drift: float = 0.0,
    vol: float = 0.006,
    seed: int = 0,
    end: str | None = None,
) -> pd.DataFrame:
    """Random-walk daily OHLC for demos and tests (no network needed)."""
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wiggle = np.abs(rng.standard_normal((2, n))) * vol * 0.6
    high = np.maximum(open_, close) * (1 + wiggle[0])
    low = np.minimum(open_, close) * (1 - wiggle[1])
    idx = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=n, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


# ------------------------------------------------- ad-hoc (non-ledger) usage

def cache_path(cache_dir: Path, pair: Pair) -> Path:
    return Path(cache_dir) / f"{pair.code}.csv"


def load_prices(pair: Pair, cache_dir: Path | str = "data/cache", offline: bool = False,
                years: int = 5) -> tuple[pd.DataFrame, str]:
    """Daily prices for one-off terminal forecasts; cached as CSV."""
    cache_dir = Path(cache_dir)
    path = cache_path(cache_dir, pair)
    if not offline:
        try:
            df, source = fetch_daily(pair, years=years)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index_label="date", float_format="%.6f")
            return df, source
        except Exception:
            if not path.exists():
                raise
    if not path.exists():
        raise RuntimeError(f"{pair.code}: no cached data in {cache_dir}; run without --offline first")
    return _clean_daily(pd.read_csv(path, index_col="date", parse_dates=["date"])), "cache"
