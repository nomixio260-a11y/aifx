"""Short-term interest rates, stored point-in-time for the carry rules.

Once a day the server downloads the FRED series of research (history.py)
and stores their recent history as one item in ``rates/``. What a rule may
use at a given day follows the same publication lags as the research: a
monthly average only from two months after the month starts, a daily rate
(averaged over a trailing month, because overnight fixings jump at quarter
ends) from the next day.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from .history import RATE_SERIES, fetch_fred

KEEP_DAYS = 460          # daily history kept in each item (a year of backtest plus the trailing mean)
KEEP_MONTHS = 30


def collect_rates(at: datetime) -> tuple[dict | None, str | None]:
    """Today's rates item: the recent history of every series (or an error)."""
    series, errors = {}, []
    for cur, items in RATE_SERIES.items():
        for sid, freq in items:
            try:
                s = fetch_fred(sid)
            except Exception as exc:          # one missing series only disables that currency's carry
                errors.append(f"{sid}: {exc}")
                continue
            since = at.replace(tzinfo=None) - (timedelta(days=KEEP_DAYS) if freq == "d" else timedelta(days=31 * KEEP_MONTHS))
            s = s[s.index >= since]
            series[sid] = [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in s.items()]
    if not series:
        return None, "rates: " + "; ".join(errors)
    day = at.strftime("%Y-%m-%d")
    return {"id": f"rates-{day}", "date": day, "series": series}, ("rates: " + "; ".join(errors)) if errors else None


def _known_series(item: dict, sid: str, freq: str) -> pd.Series | None:
    rows = item.get("series", {}).get(sid)
    if not rows:
        return None
    s = pd.Series([v for _, v in rows], index=pd.DatetimeIndex([d for d, _ in rows]), dtype="float64")
    if freq == "m":
        s.index = s.index + pd.DateOffset(months=2)
    else:
        s = s.rolling(21, min_periods=5).mean()
        s.index = s.index + pd.Timedelta(days=1)
    return s.dropna()


def known_rate(item: dict | None, cur: str, day: date | datetime | str) -> float | None:
    """The short rate of ``cur`` (% a year) as it could be known on ``day``."""
    if not item:
        return None
    day = pd.Timestamp(day).tz_localize(None) if pd.Timestamp(day).tzinfo else pd.Timestamp(day)
    value = None
    for sid, freq in RATE_SERIES.get(cur, []):     # later series take over where they have data
        s = _known_series(item, sid, freq)
        if s is None:
            continue
        s = s[s.index <= day.normalize()]
        if len(s):
            value = float(s.iloc[-1])
    return value


def rate_diff(item: dict | None, base: str, quote: str, day) -> float | None:
    """Base minus quote short rate (% a year) known on ``day``."""
    a, b = known_rate(item, base, day), known_rate(item, quote, day)
    return None if a is None or b is None else round(a - b, 4)


def latest(items: list[dict]) -> dict | None:
    return max(items, key=lambda it: (it["date"], it.get("fetched_at", ""))) if items else None
