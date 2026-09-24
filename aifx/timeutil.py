"""UTC helpers and the FX trading calendar.

The spot FX market trades continuously from Sunday 17:00 to Friday 17:00
New York time. Hourly forecast targets are counted in *open-market* hours
so a "24h ahead" forecast issued on Friday afternoon lands on Monday, not
in the middle of the weekend closure. Daily bars follow Yahoo's convention:
one bar per London calendar day, complete at London midnight.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
TOKYO = ZoneInfo("Asia/Tokyo")
HOUR = timedelta(hours=1)


def utcnow() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(t: datetime) -> str:
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def floor_hour(t: datetime) -> datetime:
    return t.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def is_market_open(t: datetime) -> bool:
    """Whether spot FX is trading at instant ``t``."""
    ny = t.astimezone(NEW_YORK)
    wd = ny.weekday()  # Monday=0 .. Sunday=6
    if wd == 5:
        return False
    if wd == 6:
        return ny.hour >= 17
    if wd == 4:
        return ny.hour < 17
    return True


def add_trading_hours(t: datetime, n: int) -> datetime:
    """End of the ``n``-th open-market hour after ``t`` (``t`` on an hour boundary)."""
    cur = t.astimezone(UTC)
    left = n
    while left > 0:
        if is_market_open(cur):
            left -= 1
        cur += HOUR
    return cur


def trading_hours_between(start: datetime, end: datetime) -> list[datetime]:
    """Open-market hour slots (their start times) in [start, end)."""
    out = []
    cur = start.astimezone(UTC)
    while cur < end:
        if is_market_open(cur):
            out.append(cur)
        cur += HOUR
    return out


def london_day_end(d: date) -> datetime:
    """London midnight at the end of calendar day ``d``, in UTC."""
    return datetime.combine(d + timedelta(days=1), time(0), tzinfo=LONDON).astimezone(UTC)


def london_date(t: datetime) -> date:
    return t.astimezone(LONDON).date()


def add_business_days(d: date, n: int) -> date:
    cur = d
    left = n
    while left > 0:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            left -= 1
    return cur


def jst(t: datetime) -> str:
    return t.astimezone(TOKYO).strftime("%Y-%m-%d %H:%M JST")
