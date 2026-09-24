import pandas as pd

from aifx.data import get_pair, parse_frankfurter, parse_yahoo_chart, synthetic_prices


def test_get_pair_accepts_common_spellings():
    assert get_pair("usd/jpy").code == "USDJPY"
    assert get_pair("EUR_USD").decimals == 5


def test_yahoo_bars_use_exchange_calendar_date():
    # Yahoo stamps daily FX bars at London midnight: 23:00 UTC during BST.
    payload = {"chart": {"result": [{
        "meta": {"exchangeTimezoneName": "Europe/London"},
        "timestamp": [1789945200, 1790031600, 1790118000],  # 2026-09-20/21/22 23:00 UTC
        "indicators": {"quote": [{
            "open": [157.0, 157.3, None],
            "high": [157.5, 157.8, 158.0],
            "low": [156.8, 157.1, 157.2],
            "close": [157.3, 157.5, None],
        }]},
    }]}}
    df = parse_yahoo_chart(payload)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-09-21", "2026-09-22"]  # NaN close dropped
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()


def test_frankfurter_rates_become_flat_bars():
    payload = {"rates": {"2026-09-01": {"JPY": 160.16}, "2026-09-02": {"JPY": 159.6}}}
    df = parse_frankfurter(payload, "JPY")
    assert df.loc["2026-09-02", "close"] == 159.6
    assert df.loc["2026-09-02", "open"] == df.loc["2026-09-02", "high"] == 159.6


def test_synthetic_prices_are_weekday_ohlc():
    df = synthetic_prices(n=100, seed=1)
    assert len(df) == 100 and (df.index.dayofweek < 5).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert isinstance(df.index, pd.DatetimeIndex)
