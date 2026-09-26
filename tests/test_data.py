from datetime import datetime, timedelta, timezone

import pandas as pd

from aifx.data import (PAIRS, SyntheticMarket, get_pair, parse_frankfurter, parse_yahoo_chart,
                       parse_yahoo_intraday, synthetic_prices)

UTC = timezone.utc


def test_get_pair_accepts_common_spellings():
    assert get_pair("usd/jpy").code == "USDJPY"
    assert get_pair("EUR_USD").decimals == 5


def test_yahoo_daily_bars_use_exchange_calendar_date():
    # Yahoo stamps daily FX bars at London midnight: 23:00 UTC during BST.
    payload = {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "Europe/London"},
        "timestamp": [1789945200, 1790031600, 1790118000],  # 2026-09-20/21/22 23:00 UTC
        "indicators": {"quote": [{
            "open": [157.0, 157.3, None], "high": [157.5, 157.8, 158.0],
            "low": [156.8, 157.1, 157.2], "close": [157.3, 157.5, None],
        }]},
    }]}}
    df = parse_yahoo_chart(payload)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-09-21", "2026-09-22"]  # NaN close dropped
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()


def test_intraday_keeps_only_completed_aligned_bars():
    base = int(datetime(2026, 9, 24, 3, tzinfo=UTC).timestamp())
    ts = [base, base + 3600, base + 7200, base + 7200 + 1311]  # 03:00, 04:00, 05:00 (forming), live point
    payload = {"chart": {"result": [{
        "meta": {"regularMarketPrice": 158.0, "regularMarketTime": ts[-1]},
        "timestamp": ts,
        "indicators": {"quote": [{"open": [1, 2, 3, 4], "high": [1, 2, 3, 4], "low": [1, 2, 3, 4], "close": [1, 2, 3, 4]}]},
    }]}}
    cutoff = datetime(2026, 9, 24, 5, 21, tzinfo=UTC)
    df, live = parse_yahoo_intraday(payload, cutoff)
    assert [t.hour for t in df.index] == [3, 4]
    assert live["price"] == 158.0


def test_frankfurter_rates_become_flat_bars():
    payload = {"rates": {"2026-09-01": {"JPY": 160.16}, "2026-09-02": {"JPY": 159.6}}}
    df = parse_frankfurter(payload, "JPY")
    assert df.loc["2026-09-02", "close"] == 159.6
    assert df.loc["2026-09-02", "open"] == df.loc["2026-09-02", "high"] == 159.6


def test_synthetic_market_bars_never_change_once_they_exist():
    m = SyntheticMarket(seed=2, start=datetime(2026, 8, 1, tzinfo=UTC))
    a, _ = m.hourly(PAIRS["USDJPY"], datetime(2026, 9, 10, tzinfo=UTC))
    b, _ = m.hourly(PAIRS["USDJPY"], datetime(2026, 9, 20, tzinfo=UTC))
    assert len(b) > len(a)
    pd.testing.assert_frame_equal(a, b.loc[a.index])
    assert not any(t.weekday() == 5 for t in b.index)  # no Saturday bars


def test_synthetic_daily_prices_are_weekday_ohlc():
    df = synthetic_prices(n=100, seed=1)
    assert len(df) == 100 and (df.index.dayofweek < 5).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert isinstance(df.index, pd.DatetimeIndex)
    assert timedelta(days=1)


def test_daily_bars_are_rebuilt_for_the_london_day():
    import numpy as np

    from aifx.data import london_days
    # hourly bars Monday 2026-09-14 ... Wednesday (British summer time: the London day ends at 23:00 UTC)
    idx = pd.date_range("2026-09-13 21:00", "2026-09-16 22:00", freq="h", tz="UTC")
    close = 150 + np.arange(len(idx)) * 0.01
    hourly = pd.DataFrame({"open": close - 0.005, "high": close + 0.02, "low": close - 0.02, "close": close}, index=idx)
    # Yahoo-style daily bars: the close is the price at 00:00 UTC at the start of the day
    days = pd.DatetimeIndex(["2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16"])
    start = [149.0, 150.02, 150.26, 150.50]
    daily = pd.DataFrame({"open": start, "high": [149.5, 150.3, 150.6, 150.8], "low": [148.9, 150.0, 150.2, 150.4],
                          "close": start}, index=days)
    out = london_days(daily, hourly)
    ends = hourly.index + pd.Timedelta(hours=1)
    mon = hourly[(ends > pd.Timestamp("2026-09-11 23:00", tz="UTC")) & (ends <= pd.Timestamp("2026-09-14 23:00", tz="UTC"))]
    assert out.loc["2026-09-14", "close"] == mon["close"].iloc[-1]          # the hourly close at 23:00 UTC
    assert out.loc["2026-09-14", "open"] == mon["open"].iloc[0]             # Sunday evening counts towards Monday
    assert out.loc["2026-09-14", "high"] == mon["high"].max() and out.loc["2026-09-14", "low"] == mon["low"].min()
    assert out.loc["2026-09-11", "close"] == 150.02                          # before the hourly bars: the next open
    # a day rebuilt from hourly bars only uses the bars of that day
    later = hourly[hourly.index < pd.Timestamp("2026-09-15 12:00", tz="UTC")]
    assert london_days(daily.loc[:"2026-09-14"], later).loc["2026-09-14"].equals(out.loc["2026-09-14"])
