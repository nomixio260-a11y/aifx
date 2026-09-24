from datetime import date, datetime, timezone

from aifx.timeutil import (add_business_days, add_trading_hours, is_market_open, london_day_end,
                           trading_hours_between)

UTC = timezone.utc


def test_market_hours_follow_new_york_close():
    # 2026-09-25 is a Friday; New York is on EDT (UTC-4), so 17:00 NY = 21:00 UTC.
    assert is_market_open(datetime(2026, 9, 25, 20, 59, tzinfo=UTC))
    assert not is_market_open(datetime(2026, 9, 25, 21, 0, tzinfo=UTC))
    assert not is_market_open(datetime(2026, 9, 26, 12, 0, tzinfo=UTC))       # Saturday
    assert not is_market_open(datetime(2026, 9, 27, 20, 59, tzinfo=UTC))      # Sunday before open
    assert is_market_open(datetime(2026, 9, 27, 21, 0, tzinfo=UTC))           # Sunday 17:00 NY


def test_market_hours_in_winter_time():
    # 2026-12-04 is a Friday; EST (UTC-5) so the close is 22:00 UTC.
    assert is_market_open(datetime(2026, 12, 4, 21, 30, tzinfo=UTC))
    assert not is_market_open(datetime(2026, 12, 4, 22, 0, tzinfo=UTC))


def test_trading_hours_skip_the_weekend():
    fri = datetime(2026, 9, 25, 20, tzinfo=UTC)
    assert add_trading_hours(fri, 1) == datetime(2026, 9, 25, 21, tzinfo=UTC)
    assert add_trading_hours(fri, 2) == datetime(2026, 9, 27, 22, tzinfo=UTC)
    slots = trading_hours_between(fri, datetime(2026, 9, 28, 0, tzinfo=UTC))
    assert len(slots) == 1 + 3  # Friday 20:00, then Sunday 21:00-24:00


def test_london_day_end_and_business_days():
    assert london_day_end(date(2026, 9, 24)) == datetime(2026, 9, 24, 23, tzinfo=UTC)   # BST
    assert london_day_end(date(2026, 12, 24)) == datetime(2026, 12, 25, 0, tzinfo=UTC)  # GMT
    assert add_business_days(date(2026, 9, 25), 1) == date(2026, 9, 28)
    assert add_business_days(date(2026, 9, 25), 5) == date(2026, 10, 2)
