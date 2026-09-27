"""Settlement calendars and spot value dates of the 7 pairs' currencies, by rule (research/rollover.md).

At the 17:00 New York roll the spot value date moves on; how many days it moves (1, or 3 over the
weekend, more around holidays) sets how far the quotes shift by the interest-rate difference. The
calendars are the Federal Reserve's, TARGET2's, England's, Sydney's and Japan's bank holidays, computed
by rule (with the moved and one-off days of recent years); a new one-off holiday has to be added when
it is announced.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

D1 = timedelta(days=1)


def _easter(y: int) -> date:
    """Easter Sunday (Gregorian)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m7 = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * m7) // 451
    return date(y, (h + m7 - 7 * m + 114) // 31, (h + m7 - 7 * m + 114) % 31 + 1)


def _nth(y: int, month: int, weekday: int, n: int) -> date:
    """The n-th ``weekday`` (0 = Monday) of the month; n = -1 is the last."""
    if n > 0:
        d = date(y, month, 1)
        return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
    d = (date(y + 1, 1, 1) if month == 12 else date(y, month + 1, 1)) - D1
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _moved(days: list[date]) -> set[date]:
    """Holidays that fall on a weekend move to the next weekday that is not already one."""
    out = {d for d in days if d.weekday() < 5}
    for d in sorted(x for x in days if x.weekday() >= 5):
        n = d
        while n.weekday() >= 5 or n in out:
            n += D1
        out.add(n)
    return out


def _usd(y: int) -> set[date]:
    """Federal Reserve holidays (a Sunday holiday moves to Monday, a Saturday one is not observed)."""
    fixed = [date(y, 1, 1), date(y, 7, 4), date(y, 11, 11), date(y, 12, 25)] + ([date(y, 6, 19)] if y >= 2022 else [])
    out = {d + D1 if d.weekday() == 6 else d for d in fixed if d.weekday() != 5}
    return out | {_nth(y, 1, 0, 3), _nth(y, 2, 0, 3), _nth(y, 5, 0, -1), _nth(y, 9, 0, 1), _nth(y, 10, 0, 2),
                  _nth(y, 11, 3, 4)}


def _eur(y: int) -> set[date]:
    """TARGET2 closing days."""
    e = _easter(y)
    return {date(y, 1, 1), e - 2 * D1, e + D1, date(y, 5, 1), date(y, 12, 25), date(y, 12, 26)}


_GBP_SPECIAL = {2011: [date(2011, 4, 29)], 2012: [date(2012, 6, 5)], 2022: [date(2022, 6, 3), date(2022, 9, 19)],
                2023: [date(2023, 5, 8)]}


def _gbp(y: int) -> set[date]:
    """Bank holidays in England (with the moved and extra days of 2011-2023)."""
    e = _easter(y)
    early = date(2020, 5, 8) if y == 2020 else _nth(y, 5, 0, 1)
    spring = {2012: date(2012, 6, 4), 2022: date(2022, 6, 2)}.get(y, _nth(y, 5, 0, -1))
    return (_moved([date(y, 1, 1), date(y, 12, 25), date(y, 12, 26)])
            | {e - 2 * D1, e + D1, early, spring, _nth(y, 8, 0, -1)} | set(_GBP_SPECIAL.get(y, [])))


def _aud(y: int) -> set[date]:
    """Bank holidays in Sydney."""
    e = _easter(y)
    out = _moved([date(y, 1, 1), date(y, 1, 26), date(y, 12, 25), date(y, 12, 26)])
    out |= {e - 2 * D1, e + D1, date(y, 4, 25), _nth(y, 6, 0, 2), _nth(y, 8, 0, 1), _nth(y, 10, 0, 1)}
    return out | ({date(2022, 9, 22)} if y == 2022 else set())


def _equinox(y: int, base: float) -> int:
    return int(base + 0.242194 * (y - 1980) - (y - 1980) // 4)


def _jpy(y: int) -> set[date]:
    """Japanese bank holidays: national holidays (a weekday between two holidays is one; a Sunday holiday
    moves to the next free day), plus 31 December and 2-3 January."""
    nat = {date(y, 1, 1), _nth(y, 1, 0, 2), date(y, 2, 11), date(y, 3, _equinox(y, 20.8431)), date(y, 4, 29),
           date(y, 5, 3), date(y, 5, 5), _nth(y, 9, 0, 3), date(y, 9, _equinox(y, 23.2488)), date(y, 11, 3),
           date(y, 11, 23)}
    nat |= {date(y, 12, 23)} if y <= 2018 else set()
    nat |= {date(y, 2, 23)} if y >= 2020 else set()
    nat |= {date(y, 5, 4)} if y >= 2007 else set()
    olympic = {2020: (date(2020, 7, 23), date(2020, 7, 24), date(2020, 8, 10)),
               2021: (date(2021, 7, 22), date(2021, 7, 23), date(2021, 8, 8))}
    if y in olympic:
        nat |= set(olympic[y])
    else:
        nat |= {_nth(y, 7, 0, 3), _nth(y, 10, 0, 2)} | ({date(y, 8, 11)} if y >= 2016 else set())
    nat |= {date(2019, 5, 1), date(2019, 10, 22)} if y == 2019 else set()
    for d in sorted(nat):
        if d + 2 * D1 in nat and d + D1 not in nat and (d + D1).weekday() != 6:
            nat.add(d + D1)
    for d in sorted(x for x in nat if x.weekday() == 6):
        n = d + D1
        while n in nat:
            n += D1
        nat.add(n)
    return nat | {date(y, 1, 2), date(y, 1, 3), date(y, 12, 31)}


CALENDARS = {"USD": _usd, "EUR": _eur, "GBP": _gbp, "AUD": _aud, "JPY": _jpy}


def holidays(cur: str, y0: int, y1: int) -> set[date]:
    return set().union(*(CALENDARS[cur](y) for y in range(y0, y1 + 1)))


def spot_date(trade: date, base: str, quote: str, hol: dict[str, set[date]]) -> date:
    """Spot value date of a trade date: T+2 business days. T+1 must be a business day of the non-USD
    currencies (a USD holiday does not count there), the value date one of both currencies and USD."""
    other = [c for c in (base, quote) if c != "USD"]
    d = trade + D1
    while d.weekday() >= 5 or any(d in hol[c] for c in other):
        d += D1
    d += D1
    while d.weekday() >= 5 or any(d in hol[c] for c in (base, quote, "USD")):
        d += D1
    return d


@lru_cache(maxsize=64)
def _year_holidays(cur: str, year: int) -> frozenset:
    return frozenset(holidays(cur, year - 1, year + 1))


def roll_days(trade: date, base: str, quote: str) -> int:
    """Value days the 17:00 New York roll at the end of weekday ``trade`` moves the spot date on."""
    hol = {c: _year_holidays(c, trade.year) for c in {base, quote, "USD"}}
    nxt = trade + timedelta(days=3 if trade.weekday() == 4 else 1)
    return (spot_date(nxt, base, quote, hol) - spot_date(trade, base, quote, hol)).days
