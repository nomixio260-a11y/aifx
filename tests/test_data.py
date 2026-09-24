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
