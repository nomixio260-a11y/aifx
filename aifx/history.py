"""Long price and interest-rate history for research (not part of the ledger).

Daily bars come from Yahoo Finance by explicit date range (the "max" range
returns thinned monthly points), hourly bars cover Yahoo's 730-day limit, and
short-term interest rates come from FRED. Rates are made point-in-time: a
monthly average is only usable two months after the month starts (it is
published after the month ends), a daily rate from the next day.
"""

from __future__ import annotations

import io
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PAIRS, Pair, _clean_daily, _yahoo_chart, http_get, parse_yahoo_chart, parse_yahoo_intraday
from .timeutil import utcnow

HIST_DIR = Path("data/history")

# currency -> list of (FRED series id, frequency); later entries take over where they have data
RATE_SERIES = {
    "USD": [("DTB3", "d")],
    "JPY": [("IRSTCI01JPM156N", "m")],
    "EUR": [("IR3TIB01EZM156N", "m"), ("ECBESTRVOLWGTTRMDMNRT", "d")],
    "GBP": [("IR3TIB01GBM156N", "m"), ("IUDSOIA", "d")],
    "AUD": [("IR3TIB01AUM156N", "m")],
}


# stock index per currency (research on month-end hedging and risk appetite)
EQUITY = {"USD": "^GSPC", "JPY": "^N225", "EUR": "^GDAXI", "GBP": "^FTSE", "AUD": "^AXJO"}
FUTURES = {"ES": "ES=F"}             # S&P 500 futures, hourly (trade nearly around the clock)
EXTRA_RATES = ("DGS2",)              # US 2-year Treasury yield, daily


def _path(name: str, root: Path | None = None) -> Path:
    return (root or HIST_DIR) / name


def fetch_daily_history(pair: Pair, start: str = "2000-01-01") -> pd.DataFrame:
    p1 = int(datetime.fromisoformat(start).timestamp())
    q = f"period1={p1}&period2={int(time.time())}&interval=1d"
    return parse_yahoo_chart(_yahoo_chart(pair.yahoo_symbol, q))


def fetch_hourly_history(pair: Pair) -> pd.DataFrame:
    df, _ = parse_yahoo_intraday(_yahoo_chart(pair.yahoo_symbol, "range=730d&interval=60m"), utcnow())
    return df


INTRADAY = {"15m": 15, "5m": 5}      # Yahoo keeps about 60 days of these


def fetch_intraday_history(pair: Pair, interval: str) -> pd.DataFrame:
    df, _ = parse_yahoo_intraday(_yahoo_chart(pair.yahoo_symbol, f"range=60d&interval={interval}"), utcnow(),
                                 minutes=INTRADAY[interval])
    return df


def fetch_fred(series: str) -> pd.Series:
    raw = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + urllib.parse.quote(series), timeout=40)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"]


def download(root: Path | None = None, log=print) -> None:
    root = root or HIST_DIR
    root.mkdir(parents=True, exist_ok=True)
    for code, pair in PAIRS.items():
        d = fetch_daily_history(pair)
        d.to_csv(_path(f"{code}_1d.csv", root), index_label="date", float_format="%.6f")
        h = fetch_hourly_history(pair)
        h.to_csv(_path(f"{code}_1h.csv", root), index_label="time", float_format="%.6f")
        log(f"{code}: {len(d)} daily bars from {d.index[0].date()}, {len(h)} hourly bars from {h.index[0]}")
        for interval in INTRADAY:
            m = fetch_intraday_history(pair, interval)
            m.to_csv(_path(f"{code}_{interval}.csv", root), index_label="time", float_format="%.6f")
            log(f"{code}: {len(m)} {interval} bars from {m.index[0]}")
    for cur, series in RATE_SERIES.items():
        for sid, _freq in series:
            fetch_fred(sid).to_csv(_path(f"rate_{sid}.csv", root), index_label="date")
            log(f"rate {cur} {sid}")
    fetch_fred("VIXCLS").to_csv(_path("rate_VIXCLS.csv", root), index_label="date")
    log("VIX (VIXCLS)")
    download_extra(root, log)


def download_extra(root: Path | None = None, log=print) -> None:
    """Stock indices, S&P 500 futures and the US 2-year yield (research_direction.py)."""
    root = root or HIST_DIR
    root.mkdir(parents=True, exist_ok=True)
    p1 = int(datetime(2000, 1, 1).timestamp())
    for cur, sym in EQUITY.items():
        d = parse_yahoo_chart(_yahoo_chart(sym, f"period1={p1}&period2={int(time.time())}&interval=1d"))
        d.to_csv(_path(f"eq_{cur}_1d.csv", root), index_label="date", float_format="%.6f")
        log(f"equity {cur} {sym}: {len(d)} days from {d.index[0].date()}")
    for key, sym in FUTURES.items():
        h, _ = parse_yahoo_intraday(_yahoo_chart(sym, "range=730d&interval=60m"), utcnow())
        h.to_csv(_path(f"fut_{key}_1h.csv", root), index_label="time", float_format="%.6f")
        log(f"futures {key} {sym}: {len(h)} hours from {h.index[0]}")
    for sid in EXTRA_RATES:
        fetch_fred(sid).to_csv(_path(f"rate_{sid}.csv", root), index_label="date")
        log(f"rate {sid}")


def load_daily(code: str, root: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(_path(f"{code}_1d.csv", root), index_col="date", parse_dates=["date"])
    return _clean_daily(df)


def load_vix(root: Path | None = None) -> pd.Series:
    """VIX close by date, shifted two days for FRED's publication delay."""
    s = pd.read_csv(_path("rate_VIXCLS.csv", root), index_col="date", parse_dates=["date"]).iloc[:, 0]
    s.index = s.index + pd.Timedelta(days=2)
    return s[~s.index.duplicated(keep="last")]


def load_equity(cur: str, root: Path | None = None) -> pd.Series:
    """Close of the currency's stock index by local trading date."""
    s = pd.read_csv(_path(f"eq_{cur}_1d.csv", root), index_col="date", parse_dates=["date"])["close"]
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_futures_hourly(key: str, root: Path | None = None) -> pd.DataFrame:
    return _load_times(_path(f"fut_{key}_1h.csv", root))


def load_fred(sid: str, root: Path | None = None) -> pd.Series:
    s = pd.read_csv(_path(f"rate_{sid}.csv", root), index_col="date", parse_dates=["date"]).iloc[:, 0]
    return s[~s.index.duplicated(keep="last")].sort_index()


def _load_times(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df.astype("float64")


def load_hourly(code: str, root: Path | None = None) -> pd.DataFrame:
    return load_intraday(code, "1h", root)


def load_intraday(code: str, interval: str, root: Path | None = None) -> pd.DataFrame:
    return _load_times(_path(f"{code}_{interval}.csv", root))


def rates_panel(index: pd.DatetimeIndex, root: Path | None = None) -> pd.DataFrame:
    """Short rates (% p.a.) per currency as they were known on each date of ``index``."""
    out = {}
    for cur, series in RATE_SERIES.items():
        known = pd.Series(np.nan, index=index)
        for sid, freq in series:
            s = pd.read_csv(_path(f"rate_{sid}.csv", root), index_col="date", parse_dates=["date"]).iloc[:, 0]
            if freq == "m":
                s.index = s.index + pd.DateOffset(months=2)   # monthly average published after month end
            else:
                # Overnight fixings jump around quarter/year ends; a trailing month
                # average keeps the level. Each fixing is known the next day.
                s = s.rolling(21, min_periods=5).mean()
                s.index = s.index + pd.Timedelta(days=1)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            aligned = s.reindex(index.union(s.index)).ffill().reindex(index)
            known = aligned.where(aligned.notna(), known)
        out[cur] = known
    return pd.DataFrame(out, index=index)
