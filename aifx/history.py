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
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PAIRS, USER_AGENT, Pair, _clean_daily, _yahoo_chart, http_get, parse_yahoo_chart, parse_yahoo_intraday
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


# ------------------------------------------------------------------ Dukascopy
# Free hourly bid and ask candles from 2003 (datafeed.dukascopy.com, one LZMA file per month):
# about 20 years of hourly bars for research, against Yahoo's 730 days.
DUKA_URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{year}/{month:02d}/{side}_candles_hour_1.bi5"
DUKA_START = 2003
DUKA_DIR = "duka"


def _duka_month(code: str, year: int, month: int, side: str, point: float) -> pd.DataFrame:
    """One month of hourly candles (``month`` 1-12); empty if Dukascopy has none."""
    import lzma
    import urllib.error
    url = DUKA_URL.format(pair=code, year=year, month=month - 1, side=side)
    raw = b""
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=40) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return pd.DataFrame()
            time.sleep(2 ** attempt)            # 429 / 503: the feed asks callers to slow down
        except Exception:
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Dukascopy: {url} failed")
    if not raw:
        return pd.DataFrame()
    rec = np.frombuffer(lzma.decompress(raw), dtype=np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"),
                                                              ("h", ">i4"), ("v", ">f4")]))
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    df = pd.DataFrame({k: rec[k].astype(float) * point for k in ("o", "c", "l", "h")},
                      index=start + pd.to_timedelta(rec["t"].astype(np.int64), unit="s"))
    df["v"] = rec["v"].astype(float)
    return df


def fetch_dukascopy_hourly(code: str, start_year: int = DUKA_START, until: datetime | None = None,
                           workers: int = 4) -> pd.DataFrame:
    """Hourly mid bars (the average of bid and ask) with the bid-ask spread, for every complete month
    from ``start_year``. Hours without ticks (weekends, holidays) are left out."""
    from concurrent.futures import ThreadPoolExecutor
    pair = PAIRS[code]
    point = 1e-3 if pair.quote == "JPY" else 1e-5
    until = until or utcnow()
    months = [(y, m) for y in range(start_year, until.year + 1) for m in range(1, 13) if (y, m) < (until.year, until.month)]
    jobs = [(y, m, side) for y, m in months for side in ("BID", "ASK")]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(lambda j: _duka_month(code, j[0], j[1], j[2], point), jobs))
    bid = pd.concat([p for (y, m, side), p in zip(jobs, parts) if side == "BID" and len(p)])
    ask = pd.concat([p for (y, m, side), p in zip(jobs, parts) if side == "ASK" and len(p)])
    both = bid.join(ask, lsuffix="_b", rsuffix="_a", how="inner")
    both = both[(both["v_b"] > 0) & (both["v_a"] > 0)]
    out = pd.DataFrame({"open": (both["o_b"] + both["o_a"]) / 2, "high": (both["h_b"] + both["h_a"]) / 2,
                        "low": (both["l_b"] + both["l_a"]) / 2, "close": (both["c_b"] + both["c_a"]) / 2,
                        "spread": both["c_a"] - both["c_b"]}, index=both.index)
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    return out[~out.index.duplicated()].sort_index()


def download_dukascopy(root: Path | None = None, log=print, start_year: int = DUKA_START) -> None:
    root = (root or HIST_DIR) / DUKA_DIR
    root.mkdir(parents=True, exist_ok=True)
    for code in PAIRS:
        df = fetch_dukascopy_hourly(code, start_year)
        df.to_csv(root / f"{code}_1h.csv", index_label="time", float_format="%.6f")
        log(f"Dukascopy {code}: {len(df):,} hourly bars {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}")


def load_long_hourly(code: str, root: Path | None = None) -> pd.DataFrame:
    """About 20 years of hourly mid bars (Dukascopy) with the bid-ask spread."""
    return _load_times((root or HIST_DIR) / DUKA_DIR / f"{code}_1h.csv")


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
