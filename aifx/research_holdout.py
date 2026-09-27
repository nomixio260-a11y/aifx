"""Out-of-sample confirmation of the adopted rules on six hold-out pairs (research/holdout.md).

Every rule adopted from the tune/test studies was designed and chosen on the
7 live pairs (USDJPY, EURJPY, GBPJPY, AUDJPY, EURUSD, GBPUSD, AUDUSD). The
six pairs here (EURGBP, EURAUD, GBPAUD, USDCHF, USDCAD, NZDUSD) were never
used for anything; each adopted rule is replayed on them with its live
settings, and nothing is chosen or tuned on them. The tests and their pass
criteria (``PREREG``) were written before any hold-out result was computed:

1. the time-of-day calls (season.py), as research_season_long / research_rollover replicate them;
2. the rollover promotion (season.promote with the settlement calendars; CHF, CAD and NZD calendars
   are written here by rule, next to fxcalendar's USD, EUR, GBP, AUD and JPY);
3. the hourly trade rule carry_mom_vol against the retired carry_mom (research_signals' cost model);
4. the daily band shape change (engine.BAND_NU "1d", research_bands' daily pipeline);
5. Plan B of research/bands.md (k half-life per horizon), not adopted.

Data: Dukascopy hourly mid bars with the bid-ask spread from 2006 (history.load_long_hourly),
point-in-time short rates (history.rates_panel, CHF / CAD / NZD monthly 3-month interbank). Periods:
2006-2016 and 2017-. The same code is also run on the 7 live pairs: with the previous live season rule it
reproduces the published numbers (a check of this module), and with the current one it gives the
7-pair figures the hold-out ones are compared with.

    python -m aifx.research_holdout     # writes research/holdout.md and research/holdout.json
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import engine, fxcalendar, history, season, trade
from . import research_bands as rb
from . import research_rollover as ro
from . import research_signals as rs
from .data import PAIRS, Pair
from .fxcalendar import D1, _easter, _moved, _nth, spot_date
from .timeutil import add_trading_minutes, market_open_mask

REPORT_DIR = Path("research")
CODES = history.HOLDOUT_PAIRS
NAMES = {"EURGBP": "ユーロ/ポンド", "EURAUD": "ユーロ/豪ドル", "GBPAUD": "ポンド/豪ドル",
         "USDCHF": "米ドル/スイスフラン", "USDCAD": "米ドル/カナダドル", "NZDUSD": "NZドル/米ドル"}
HPAIRS = {c: Pair(c, c[:3], c[3:], NAMES[c]) for c in CODES}
SPLIT = pd.Timestamp("2017-01-01", tz="UTC")          # first half: origins (entries) before, second half: from
PERIODS = ("tune", "test")
PERIOD_JA = {"tune": "2006〜2016", "test": "2017〜"}
SEED = 11
WORKERS = 2

# ------------------------------------------------------------------ the pre-registered tests (fixed first)

HIT_TOL = 0.01                                          # test 2: one percentage point
PROMOTE = {"T": season.T_MIN, "E": season.ROLL_E_HIGH, "days": "cal"}   # live promotion (research_rollover "b")
OLD_DAILY_NU = {1: 10, 20: 30}                          # test 4: engine.BAND_NU["1d"] before research/bands.md
# the live settings when these tests were fixed (after them the rules were retired and the daily 1-day nu
# went back to 10, so the study keeps its own copy)
STUDY_RULES = {"1h": trade.TESTED["carry_mom_vol"], "1d": trade.TESTED["carry"]}
STUDY_BAND_NU = {"1h": {1: 5, 4: 5, 24: 6}, "1d": {1: 6, 5: 10, 10: 15, 20: 5}}
PLAN_B = {"half_life": {"1h": {1: 400, 4: 1600, 24: 1600}, "1d": {1: 120, 5: 120, 10: 480, 20: 480}},
          "nu": {"1h": {1: 5, 4: 5, 24: 8}, "1d": {1: 6, 5: 10, 10: 10, 20: 6}}}
PLAN_B_CHANGED = {"1h": (4, 24), "1d": (10, 20)}
EPS = 1e-12
TRADE_KEYS = ("n", "win", "pips", "total_pips", "t", "pips_live", "cost", "swap")
TEST_NO = {"calls": 1, "promote": 2, "trade": 3, "shape": 4, "planb": 5}

PREREG = """\
## 0. 事前に決めた検定 (結果を見る前に固定)

この節は、ホールドアウトのペアの結果を1つも計算する前に aifx/research_holdout.py (`PREREG`) に書いたもので、結果を見た後に変えていません。

- **対象 (ホールドアウト):** EURGBP・EURAUD・GBPAUD・USDCHF・USDCAD・NZDUSD の6ペア。本番の7ペア (USDJPY・EURJPY・GBPJPY・AUDJPY・EURUSD・GBPUSD・AUDUSD) と違い、どのルールの設計・設定の選択にも使っていません。
- **データ:** Dukascopy の1時間足 (買値と売値の仲値、各1時間の終わりのスプレッド) 2006年1月から。短期金利は history.rates_panel (FRED。CHF・CAD・NZD は月次の3か月物銀行間金利を追加し、他の月次の系列と同じく月初から2か月後に使用)。
- **期間:** 2006〜2016 (前半) と 2017〜 (後半)。1・2 は予測の起点、3 はエントリーの日付で分けます。4・5 は元の研究 (research_bands) と同じく、前半は目標時刻が2016年まで、後半は起点が2017年からです。前半の実際の始まりは、計算に必要な足がそろってからです (1・2: 6,000本 ≈ 2007年1月、3: 150本 (値動きの荒さの見送りは1,500本から)、4・5 の1時間足: 6,500本 ≈ 2007年2月、日足: 500営業日 ≈ 2008年)。
- **設定はすべて本番のまま** です (season.py・fxcalendar.py・trade.py・engine.py・learning.py の値。ホールドアウトのペアで選ぶ・調整する値はありません)。時間帯の偏りは今の本番の規則で数えます: 市場が再開した最初の足 (週末の窓開けを含む足) には方向を示さず、起点の直後に続く足がない (データが途切れた) 予測は採点しません。
- 的中率は値動きがゼロの回を除き、t は週ごと (月〜日) に全ペアの結果を合計して計算します (research_direction.score)。売買の t は月ごとの R の合計から計算します (research_signals)。

**検定と合格の条件** (6ペアをまとめて):

1. **時間帯の偏り** (season.py: 曜日×時間の枠の過去6,000本の平均、|t| ≥ 2 で方向、|t| ≥ 4 で確度: 高。research_season_long / research_rollover と同じ計算を、season.slot_stats / bar_drift と抜き取りで照合)。後半 (2017〜) で、方向を示した回の的中率が 50% を超え、t ≥ 2、かつ確度: 高 の的中率が方向を示した回全体の的中率を上回れば合格。前半も報告します。
2. **ロールオーバーの格上げ** (本番の規則、research_rollover の候補 b): ロールオーバーの足 (ニューヨーク時間17時に始まる月〜木の1時間足) で |t| ≥ 2 の方向が、見込みのずれ e = −(基準通貨の金利 − 相手通貨の金利) × 日数 / 360 × 100 と同じ向きで |e| ≥ 2 bp なら確度: 高。日数は祝日の暦と本番と同じスポット日の規則 (fxcalendar.spot_date、T+2) で数えます。CHF (チューリッヒ)・CAD (トロント)・NZD (ウェリントン・オークランド) の暦はこのモジュールに規則で書きます (「祝日の暦」の節)。前半・後半の**両方で**、格上げされた足の的中率が今の確度: 高 (|t| ≥ 4) の的中率 − 1ポイント以上、かつ格上げ後の確度: 高 の的中率が今より 1ポイントを超えて下がらなければ合格。格上げが0回の期間があれば判定不能 (合格としない)。
3. **1時間足の売買ルール:** 本番の carry_mom_vol (金利差 1%以上と過去120本の値動きが同じ向き、直近120本の対数変化の標準偏差がその前の6,000本の80%点を超えたら見送り、損切り ATR×3、利確 ATR×3、最長120本) と、以前の carry_mom (最長24本、見送りなし)。コストは research_signals と同じく「記録されたスプレッド (エントリーの足の終わりで半分、決済の足の終わりで半分、中央値の20倍で頭打ち) と本番相当のコストの大きい方」で、本番相当のコストはそのペアの記録されたスプレッドの中央値 (全期間) です。スワップは金利のパネルから (業者の取り分 0.5%/年)。前半・後半の**両方で**、carry_mom_vol のコスト・スワップ込みの1回平均 pips が carry_mom を上回れば合格。取引数・勝率・平均 pips・t を報告します。日足の金利差ルール (trade.RULES["1d"]、Dukascopy から作ったロンドン日足) も参考に報告します (合否なし)。
4. **日足のレンジの形** (engine.BAND_NU["1d"]: 1営業日先 ν 10 → 6、20営業日先 ν 30 → 5): research_bands の日足の計算 (本番の生の σ、k の学習の再現) で、**両方の予測先・両方の期間で**対数スコアが上がれば合格。
5. **案B** (research/bands.md、未採用): k の半減期を予測先ごとに {"1h": {1: 400, 4: 1600, 24: 1600}, "1d": {1: 120, 5: 120, 10: 480, 20: 480}}、ν を {"1h": {24: 8}, "1d": {10: 10, 20: 6}} (ほかは本番のまま)。変わる予測先 (1時間足の4・24時間先、日足の10・20営業日先) の**それぞれで、両方の期間で**、本番の設定より対数スコア (幅の違いを含む、値動きそのものの対数スコア) が上がり、かつ 50/80/95%レンジの的中率の名目からのずれ (3つの平均) が大きくならなければ合格。

各検定には、元の7ペアの数字を並べます (既存の報告 research/rollover.md・signals.md・bands.md の数字と、同じコードで今の規則のまま計算し直した数字)。

**補助の確認** (合否には使わない、事前に決めたもの): ペア別・年別の内訳。USD/CAD のスポットは市場の慣行では T+1 なので、2. は USDCAD を T+1 で数えた場合と USDCAD を除いた場合も示します。本番の関数との一致 (season.slot_stats / bar_drift / roll_shift / promote、trade.rule_signal、engine.sigma_steps、learning.learn_arrays)。同じコードを元の7ペアに以前の規則で当てはめ、既存の報告の数字が再現されること。
"""

# ------------------------------------------------------------------ settlement calendars of CHF, CAD, NZD

def _next_monday(d: date) -> date:
    """A Saturday or Sunday date observed on the following Monday; a weekday stays."""
    return d + timedelta(days=7 - d.weekday()) if d.weekday() >= 5 else d


def _monday_nearest(d: date) -> date:
    """The Monday nearest ``d`` (from a Friday, Saturday or Sunday the next one)."""
    wd = d.weekday()
    return d + timedelta(days=7 - wd) if wd >= 4 else d - timedelta(days=wd)


def _chf(y: int) -> set[date]:
    """Zurich bank holidays (SIX Interbank Clearing closing days): no weekend substitution."""
    e = _easter(y)
    return {date(y, 1, 1), date(y, 1, 2), e - 2 * D1, e + D1, e + 39 * D1, e + 50 * D1, date(y, 5, 1),
            date(y, 8, 1), date(y, 12, 25), date(y, 12, 26)}


def _cad(y: int) -> set[date]:
    """Toronto bank holidays (QuantLib's Canada settlement calendar): Family Day from 2008, the National
    Day for Truth and Reconciliation from 2021; fixed dates on a weekend are observed on the Monday
    (Christmas and Boxing Day on the next free weekdays)."""
    e = _easter(y)
    may24 = date(y, 5, 24)
    out = {_next_monday(date(y, 1, 1)), e - 2 * D1, may24 - timedelta(days=may24.weekday()),
           _next_monday(date(y, 7, 1)), _nth(y, 8, 0, 1), _nth(y, 9, 0, 1), _nth(y, 10, 0, 2),
           _next_monday(date(y, 11, 11))} | _moved([date(y, 12, 25), date(y, 12, 26)])
    if y >= 2008:
        out.add(_nth(y, 2, 0, 3))
    if y >= 2021:
        out.add(_next_monday(date(y, 9, 30)))
    return out


_NZD_SPECIAL = {2022: [date(2022, 6, 24), date(2022, 9, 26)], 2023: [date(2023, 7, 14)], 2024: [date(2024, 6, 28)],
                2025: [date(2025, 6, 20)], 2026: [date(2026, 7, 10)]}     # Matariki; the Queen's memorial day


def _nzd(y: int) -> set[date]:
    """New Zealand bank holidays of Wellington and Auckland (both anniversary days): New Year (1 and 2
    January), Christmas and Boxing Day moved to the next free weekdays, Waitangi and ANZAC Day on the
    Monday from 2014 when they fall on a weekend, Matariki from 2022."""
    e = _easter(y)
    out = _moved([date(y, 1, 1), date(y, 1, 2), date(y, 12, 25), date(y, 12, 26)])
    out |= {_monday_nearest(date(y, 1, 22)), _monday_nearest(date(y, 1, 29)), e - 2 * D1, e + D1,
            _nth(y, 6, 0, 1), _nth(y, 10, 0, 4)}
    out |= {(_next_monday(d) if y >= 2014 else d) for d in (date(y, 2, 6), date(y, 4, 25))}
    return out | set(_NZD_SPECIAL.get(y, []))


CALENDARS = {**fxcalendar.CALENDARS, "CHF": _chf, "CAD": _cad, "NZD": _nzd}


def holidays(cur: str, y0: int, y1: int) -> set[date]:
    """fxcalendar.holidays with the CHF, CAD and NZD calendars added."""
    return set().union(*(CALENDARS[cur](y) for y in range(y0, y1 + 1)))


def spot_next_day(trade_date: date, base: str, quote: str, hol: dict[str, set[date]]) -> date:
    """T+1 value date (the market convention of USD/CAD): the next business day of both currencies and USD."""
    d = trade_date + D1
    while d.weekday() >= 5 or any(d in hol[c] for c in (base, quote, "USD")):
        d += D1
    return d


@lru_cache(maxsize=128)
def _year_holidays(cur: str, year: int) -> frozenset:
    return frozenset(holidays(cur, year - 1, year + 1))


def roll_days(trade_date: date, base: str, quote: str, spot=spot_date) -> int:
    """fxcalendar.roll_days with these calendars."""
    hol = {c: _year_holidays(c, trade_date.year) for c in {base, quote, "USD"}}
    nxt = trade_date + timedelta(days=3 if trade_date.weekday() == 4 else 1)
    return (spot(nxt, base, quote, hol) - spot(trade_date, base, quote, hol)).days


def roll_shift(starts, base: str, quote: str, diff: float | None, spot=spot_date) -> np.ndarray:
    """season.roll_shift with these calendars (the same numbers where fxcalendar has both currencies)."""
    local = pd.DatetimeIndex(starts).tz_convert(season.NEW_YORK)
    out = np.zeros(len(local))
    if diff is None:
        return out
    for k, t in enumerate(local):
        if t.hour == 17 and t.minute == 0 and t.dayofweek <= 3:
            out[k] = -diff * roll_days(t.date(), base, quote, spot) / 360 * 100
    return out


# ------------------------------------------------------------------ 1-2: time-of-day calls and the promotion

def live_rule(F: pd.DataFrame) -> pd.DataFrame:
    """research_rollover.build rows as the live rule now scores them: no call for the first bar after the
    market reopens (season.bar_drift: its move is the weekend gap the statistics leave out), and only
    forecasts whose target bar starts where the origin ends (research_direction.session_eval)."""
    start = pd.DatetimeIndex(pd.to_datetime(F["tstart"].to_numpy(), utc=True))
    reopen = ~market_open_mask(start - pd.Timedelta(hours=1))
    G = F.copy()
    G.loc[reopen, ["d0", "t0"]] = 0.0
    return G[(G["tstart"] == G["origin"]).to_numpy()].reset_index(drop=True)


def season_frame(codes, pairs: dict | None, log=print, current: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    """Forecast rows of every pair (research_rollover.build, Dukascopy, 6,000 open bars before the first
    origin), the live rule applied when ``current``; USDCAD also with T+1 value dates (e_t1). Returns the
    rows, the check against season.slot_stats / bar_drift, and the raw bars."""
    parts, checks, raw = [], {}, {}
    for code in codes:
        H = history.load_long_hourly(code)
        H = H[~H.index.duplicated()].sort_index()
        raw[code] = H
        pair = (pairs or {}).get(code)
        F = ro.build(H, code, 60, season.WINDOW, 24, pair=pair, holidays_fn=holidays if pair else None)
        if code == "USDCAD":
            T1 = ro.build(H, code, 60, season.WINDOW, 24, pair=pair, holidays_fn=holidays, spot_fn=spot_next_day)
            F["e_t1"], F["days_t1"] = T1["e_cal"].to_numpy(), T1["days_cal"].to_numpy()
        if current:
            F = live_rule(F)
            checks[code] = ro.check_against_season(H, F, 60, n_days=8, seed=SEED)
        parts.append(F)
        if log:
            log(f"season {code}: {len(F):,} rows")
    A = pd.concat(parts, ignore_index=True)
    A["period"] = np.where(A["origin"] >= SPLIT.value, "test", "tune")
    A["year"] = pd.DatetimeIndex(pd.to_datetime(A["origin"].to_numpy(), utc=True)).year
    return A, checks, raw


def _scores(G: pd.DataFrame, cand: str = "base", p: dict | None = None) -> dict:
    s, hi = ro.apply(G, cand, p or {})
    return ro.evaluate(G, s, hi)


def calls_test(A: pd.DataFrame) -> dict:
    """Test 1: all calls and the high tier (|t| >= 4) by period, pair, year and roll bar or not."""
    out: dict = {"periods": {p: _scores(A[A["period"] == p]) for p in PERIODS}}
    te = out["periods"]["test"]
    cond = {"hit": (te["all"].get("hit") or 0) > 0.5, "t": (te["all"].get("t") or 0) >= 2.0,
            "high": (te["high"].get("hit") or 0) > (te["all"].get("hit") or 1)}
    out["conditions"] = cond
    out["pass"] = all(cond.values())
    out["pairs"] = {p: {c: _scores(g) for c, g in A[A["period"] == p].groupby("pair", sort=False)} for p in PERIODS}
    out["years"] = {int(y): _scores(g) for y, g in A.groupby("year")}
    out["roll_bar"] = {p: {k: _scores(g) for k, g in A[A["period"] == p].groupby(np.where(A.loc[A["period"] == p, "roll"],
                                                                                           "roll", "other"))}
                       for p in PERIODS}
    return out


def promote_period(G: pd.DataFrame, e_col: str = "e_cal") -> dict:
    """Test 2 on one period: the high tier before and after the promotion and the promoted calls."""
    H = G if e_col == "e_cal" else G.assign(e_cal=G[e_col].to_numpy())
    base = _scores(H)
    s, hi = ro.apply(H, "b", PROMOTE)
    new = ro.evaluate(H, s, hi)
    up = ro.changes(H, s, hi)["to_high"]
    b_hit, n_hit = base["high"].get("hit"), new["high"].get("hit")
    cond = {"promoted": up["n"] > 0 and b_hit is not None and up["hit"] is not None and up["hit"] >= b_hit - HIT_TOL - EPS,
            "high": up["n"] > 0 and b_hit is not None and n_hit is not None and n_hit >= b_hit - HIT_TOL - EPS}
    return {"base": base, "new": new, "promoted": up, "conditions": cond, "pass": all(cond.values()),
            "undecided": up["n"] == 0}


def promote_test(A: pd.DataFrame) -> dict:
    """Test 2 by period, plus (supplementary) USDCAD at T+1, without USDCAD, and per pair."""
    out: dict = {"periods": {p: promote_period(A[A["period"] == p]) for p in PERIODS}}
    out["pass"] = all(out["periods"][p]["pass"] for p in PERIODS)
    out["undecided"] = any(out["periods"][p]["undecided"] for p in PERIODS)
    if "e_t1" in A:
        e_t1 = np.where(A["pair"] == "USDCAD", A["e_t1"].fillna(0.0), A["e_cal"])
        B = A.assign(e_mix=e_t1)
        out["usdcad_t1"] = {p: promote_period(B[B["period"] == p], "e_mix") for p in PERIODS}
    X = A[A["pair"] != "USDCAD"]
    out["without_usdcad"] = {p: promote_period(X[X["period"] == p]) for p in PERIODS}
    out["pairs"] = {}
    s0, h0 = ro.apply(A, "base", {})
    s, hi = ro.apply(A, "b", PROMOTE)
    f = A["f"].to_numpy()
    up = hi & ~h0 & (f != 0)
    for p in PERIODS:
        for code in A["pair"].unique():
            m = up & (A["period"] == p).to_numpy() & (A["pair"] == code).to_numpy()
            hb = h0 & (f != 0) & (A["period"] == p).to_numpy() & (A["pair"] == code).to_numpy()
            out["pairs"].setdefault(code, {})[p] = {
                "promoted": {"n": int(m.sum()), "hit": float(np.mean(np.sign(f[m]) == s[m])) if m.any() else None},
                "high": {"n": int(hb.sum()), "hit": float(np.mean(np.sign(f[hb]) == s0[hb])) if hb.any() else None}}
    # how often the calendars change the day count, and the roll bars' |e| (information)
    R = A[A["roll"].to_numpy()]
    out["roll_bars"] = {p: {"n": int((R["period"] == p).sum()),
                            "cal_differs": int(((R["period"] == p) & (R["days_cal"] != R["days_plain"])).sum()),
                            "e_ge_2": int(((R["period"] == p) & (R["e_cal"].abs() >= PROMOTE["E"])).sum())}
                        for p in PERIODS}
    return out


def promote_live_check(A: pd.DataFrame, raw: dict, n_other: int = 40) -> dict:
    """The live path at roll-bar origins (every promoted one and a sample of the others): season.slot_stats /
    bar_drift for the call, roll_shift here (compared with the live season.roll_shift where fxcalendar has both
    currencies) and season.promote, against the flags of research_rollover.apply used above."""
    rng = np.random.default_rng(SEED)
    s_b, hi_b = ro.apply(A, "b", PROMOTE)
    _, h0 = ro.apply(A, "base", {})
    n = mism = skipped = raised = e_checked = 0
    worst_e = 0.0
    for code in A["pair"].unique():
        pair = HPAIRS.get(code) or PAIRS[code]
        live_cal = all(c in fxcalendar.CALENDARS for c in (pair.base, pair.quote))
        m = ((A["pair"] == code) & A["roll"] & np.isfinite(A["diff"])).to_numpy()
        up = np.flatnonzero(m & hi_b & ~h0)
        rest = np.flatnonzero(m & ~(hi_b & ~h0))
        pick = np.concatenate([up, rng.choice(rest, size=min(n_other, len(rest)), replace=False)])
        for k in pick:
            origin = pd.Timestamp(int(A["origin"].iat[k]), tz="UTC").to_pydatetime()
            starts = [add_trading_minutes(origin, 1, 60) - timedelta(minutes=60)]
            if pd.Timestamp(starts[0]).value != int(A["tstart"].iat[k]):
                skipped += 1
                continue
            d, t = season.bar_drift(season.slot_stats(raw[code], origin, 60), starts, 60)
            diff = float(A["diff"].iat[k])
            e = roll_shift(starts, pair.base, pair.quote, diff)
            if live_cal:
                worst_e = max(worst_e, float(np.max(np.abs(e - season.roll_shift(starts, pair.base, pair.quote, diff)))))
                e_checked += 1
            t2 = season.promote(d, t, e)
            high = bool(d[0] != 0 and abs(t2[0]) >= season.T_HIGH)
            n += 1
            raised += int(high and abs(float(A["t0"].iat[k])) < season.T_HIGH)
            mism += int(high != bool(hi_b[k]) or (d[0] != 0 and np.sign(d[0]) != s_b[k]))
    return {"origins": n, "raised": raised, "mismatch": mism, "skipped": skipped, "e_vs_live_n": e_checked,
            "e_vs_live_max": worst_e}


def calendar_check() -> dict:
    """The day counts of the rolls 2006-2026 per pair: how often the calendars change them (plain 1, Wednesday
    3), and for pairs whose currencies fxcalendar has, whether roll_days here equals fxcalendar.roll_days."""
    days = pd.bdate_range("2006-01-02", "2026-12-31")
    out = {}
    for code in CODES:
        pair = HPAIRS[code]
        cal = np.array([roll_days(d.date(), pair.base, pair.quote) for d in days if d.weekday() <= 3])
        plain = np.array([3 if d.weekday() == 2 else 1 for d in days if d.weekday() <= 3])
        row = {"rolls": int(len(cal)), "differ": int(np.sum(cal != plain)),
               "counts": {int(k): int(v) for k, v in zip(*np.unique(cal, return_counts=True))}}
        if all(c in fxcalendar.CALENDARS for c in (pair.base, pair.quote)):
            row["vs_fxcalendar_mismatch"] = int(sum(roll_days(d.date(), pair.base, pair.quote)
                                                    != fxcalendar.roll_days(d.date(), pair.base, pair.quote)
                                                    for d in days if d.weekday() <= 3))
        out[code] = row
    return out


# ------------------------------------------------------------------ 3: trade rules

def trade_rules() -> dict:
    """The live hourly rule, the retired one and the live daily rule, as research_signals variants."""
    r, d = STUDY_RULES["1h"], STUDY_RULES["1d"]
    v = r["vol"]
    assert r["key"] == "carry_mom_vol" and d["key"] == "carry"
    assert (v["n"], v["window"], v["q"]) == (rs.RV_BARS[True], rs.RV_YEAR * rs.BARS_PER_DAY, 0.8)
    old = {"thr": r["thr"], "L": r["L"], **trade.RETIRED["carry_mom"]}
    assert old == {k: x for k, x in rs.STUDY_RULES["1h"].items() if k != "key"}
    return {"carry_mom_vol": ("carry_mom", {"thr": r["thr"], "L": r["L"], "sl": r["sl"], "tp": r["tp"], "hold": r["hold"]},
                              ["vol80"], "1h"),
            "carry_mom": ("carry_mom", old, [], "1h"),
            "carry_1d": ("carry", {k: d[k] for k in ("thr", "sl", "tp", "hold")}, [], "1d")}


def trade_test(codes, pairs: dict | None, holdout: bool, log=print, checks: bool = True) -> dict:
    """Test 3: every rule of trade_rules on Dukascopy hourly bars (the daily rule on London days built from them).
    On the hold-out pairs the live-like cost is each pair's median recorded spread; on the 7 pairs trade.COST_PIPS."""
    vix = rs._vix(None)
    p1h, p1d = {}, {}
    for code in codes:
        h = history.load_long_hourly(code)
        pair = (pairs or {}).get(code)
        p1h[code] = rs.prepare(code, h, True, vix, pair=pair)
        p1d[code] = rs.prepare(code, rs.london_days(h), False, vix, start=20, pair=pair)
    live_cost = {c: float(P["spread_median"]) for c, P in p1h.items()} if holdout else None
    out: dict = {"live_cost": live_cost or {c: trade.COST_PIPS[c] for c in codes},
                 "spread_median": {c: {p: float(np.median(P["spread_pips"][(P["time"] < rs.SPLIT) == (p == "tune")]))
                                       for p in PERIODS} for c, P in p1h.items()},
                 "rules": {}}
    for name, (key, rule, filters, tf) in trade_rules().items():
        pan = p1h if tf == "1h" else p1d
        times = {c: P["time"] for c, P in pan.items()}
        tr = rs.run_variant(pan, key, rule, filters, live_cost=live_cost)
        bd = rs.breakdown(tr, times)
        out["rules"][name] = {"rule": rule, "filters": filters, "tf": tf, **rs.split(tr, times),
                              "pair": bd["pair"], "block": bd["block"], "year": bd["year"]}
        if log:
            x = out["rules"][name]
            log(f"trade {name}: tune n={x['tune'].get('n')} pips={x['tune'].get('pips')} | "
                f"test n={x['test'].get('n')} pips={x['test'].get('pips')}")
    a, b = out["rules"]["carry_mom_vol"], out["rules"]["carry_mom"]
    cond = {p: bool(a[p].get("n") and b[p].get("n") and a[p]["pips"] > b[p]["pips"]) for p in PERIODS}
    out["conditions"], out["pass"] = cond, all(cond.values())
    if checks:
        out["signal_check"] = {"1h": rs.check_signals(p1h, "1h"), "1d": rs.check_signals(p1d, "1d")}
    out["data"] = {c: {"from": str(P["time"][0].date()), "to": str(P["time"][-1].date()), "bars": int(len(P["c"]))}
                   for c, P in p1h.items()}
    return out


# ------------------------------------------------------------------ 4-5: forecast ranges

def band_collect(codes, workers: int, log=print, t0: float | None = None) -> tuple[dict, dict]:
    """research_bands' per-pair raw variances and outcomes (Dukascopy only) for the hourly and daily
    timeframes, then every horizon scored (k replayed, every nu and variant) in ``workers`` processes."""
    t0 = t0 or time.time()
    data: dict = {"1h": [], "1d": []}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [(tf, code, ex.submit(fn, (code, "duka"))) for tf, fn in (("1h", rb.hourly_pair), ("1d", rb.daily_pair))
                for code in codes]
        for tf, code, fut in futs:
            data[tf].append(fut.result())
            if log:
                log(f"bands {tf} {code}: {len(data[tf][-1]['t0']):,} origins ({time.time() - t0:.0f}s)")
    jobs, checks, ranges = [], {}, {}
    for tf, parts in data.items():
        names = list(rb.H_VARIANTS if tf == "1h" else rb.D_VARIANTS)
        P = rb.pooled(parts, names)
        checks[tf] = {p["pair"]: p["check"] for p in parts}
        ranges[tf] = [str(pd.Timestamp(int(P["t0"].min()), tz="UTC").date()), str(pd.Timestamp(int(P["t0"].max()), tz="UTC").date())]
        for j, h in enumerate(engine.TIMEFRAMES[tf].horizons):
            jobs.append((tf, "duka", h, rb.horizon_arrays(P, tf, j)))
    del data
    jobs.sort(key=lambda job: -len(job[3]["t0"]))
    res: dict = {"1h": {}, "1d": {}}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for tf, _src, h, hres, _keep in ex.map(rb.analyse_h, jobs):
            res[tf][str(h)] = {"n": hres["n"], "periods": hres["periods"],
                               "variants": {k: v for k, v in hres["variants"].items() if k in ("base", "hl4.0")}}
            if log:
                log(f"bands {tf} h={h}: {hres['n']:,} forecasts ({time.time() - t0:.0f}s)")
    return res, {"checks": checks, "ranges": ranges}


def _ls(sc: dict, nu: int, full: bool) -> float:
    return sc["nu"][str(nu)]["ls"] + (sc["jac"] if full else 0.0)


def shape_test(tfres: dict) -> dict:
    """Test 4 on research_bands' per-horizon results ``{tf: {h: hres}}``: the daily 1 and 20 day shapes,
    the old nu against the live one (the same k, so the log score of z)."""
    out: dict = {}
    for h, old in OLD_DAILY_NU.items():
        new = STUDY_BAND_NU["1d"][h]
        base = tfres["1d"][str(h)]["variants"]["base"]
        row: dict = {"old": old, "new": new}
        for p in PERIODS:
            sc = base["scores"][p]
            a, b = sc["nu"][str(old)], sc["nu"][str(new)]
            pv = base.get("dm_vs_current", {}).get(str(new), {}).get(p) if rb.BAND_NU["1d"][h] == old else None
            row[p] = {"n": sc["n"], "ls_diff": b["ls"] - a["ls"], "cover_old": a["cover"], "cover_new": b["cover"],
                      "cov_diff": rb._cover_dist(b["cover"]) - rb._cover_dist(a["cover"]), "p": pv,
                      "pass": b["ls"] - a["ls"] > EPS}
        out[str(h)] = row
    out["pass"] = all(out[str(h)][p]["pass"] for h in OLD_DAILY_NU for p in PERIODS)
    return out


def planb_test(tfres: dict) -> dict:
    """Test 5: Plan B (k half-life and nu per horizon) against the live settings at each horizon it changes:
    the log score of the move (with -log sigma, as k differs) and the mean distance of the 50/80/95 %
    coverage from nominal."""
    out: dict = {}
    for tf, hs in PLAN_B_CHANGED.items():
        for h in hs:
            mult = PLAN_B["half_life"][tf][h] / engine.TIMEFRAMES[tf].half_life
            name = "base" if mult == 1 else f"hl{mult}"
            v = tfres[tf][str(h)]["variants"]
            ref_nu, new_nu = STUDY_BAND_NU[tf][h], PLAN_B["nu"][tf][h]
            row: dict = {"half_life": [engine.TIMEFRAMES[tf].half_life, PLAN_B["half_life"][tf][h]], "nu": [ref_nu, new_nu],
                         "variant": name}
            for p in PERIODS:
                r, n = v["base"]["scores"][p], v[name]["scores"][p]
                cr, cn = r["nu"][str(ref_nu)]["cover"], n["nu"][str(new_nu)]["cover"]
                ls_d = _ls(n, new_nu, True) - _ls(r, ref_nu, True)
                cov_d = rb._cover_dist(cn) - rb._cover_dist(cr)
                row[p] = {"n": r["n"], "ls_diff": ls_d, "cover_ref": cr, "cover_new": cn, "cov_ref": rb._cover_dist(cr),
                          "cov_new": rb._cover_dist(cn), "cov_diff": cov_d,
                          "k_ref": v["base"]["k_median"][p], "k_new": v[name]["k_median"][p],
                          "ls_ok": ls_d > EPS, "cov_ok": cov_d <= EPS, "pass": ls_d > EPS and cov_d <= EPS}
            row["pass"] = all(row[p]["pass"] for p in PERIODS)
            out[f"{tf}_{h}"] = row
    out["pass"] = all(x["pass"] for x in out.values() if isinstance(x, dict))
    return out


def _band_res_of(bands_json: dict) -> dict:
    return {tf: bands_json["tf"][tf]["duka"]["h"] for tf in ("1h", "1d")}


# ------------------------------------------------------------------ the 7 pairs: published and recomputed

def _load(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def published() -> dict:
    """The 7 pairs' numbers of each test in the existing reports (rollover.json, signals.json, bands.json)."""
    rol, sig, bnd = _load("rollover.json"), _load("signals.json"), _load("bands.json")
    b = rol["candidates"]["b"]
    t1 = {p: {k: {q: rol["baseline"]["duka"][p]["base"][k].get(q) for q in ("n", "hit", "t")} for k in ("all", "high")}
          for p in PERIODS}
    t2 = {p: {"base": b["duka"][p]["base"]["high"], "new": b["duka"][p]["new"]["high"],
              "promoted": b["duka"][p]["changes"]["to_high"]} for p in PERIODS}
    v1h, v1d = sig["1h"]["variants"], sig["1d"]["variants"]
    t3 = {"carry_mom_vol": {p: {k: v1h["combo"]["by_source"]["duka_1h"][p].get(k) for k in TRADE_KEYS} for p in PERIODS},
          "carry_mom": {p: {k: v1h["base"]["by_source"]["duka_1h"][p].get(k) for k in TRADE_KEYS} for p in PERIODS},
          "carry_1d": {p: {k: v1d["base"]["by_source"]["duka_1d"][p].get(k) for k in TRADE_KEYS} for p in PERIODS}}
    assert v1h["combo"]["rule"]["hold"] == STUDY_RULES["1h"]["hold"] and v1h["combo"]["filters"] == ["vol80"]
    tfres = _band_res_of(bnd)
    return {"calls": t1, "promote": t2, "trade": t3, "shape": shape_test(tfres), "planb": planb_test(tfres),
            "spans": {"duka": rol["sources"]["duka"]["span"], "bands_1h": bnd["tf"]["1h"]["duka"]["range"],
                      "bands_1d": bnd["tf"]["1d"]["duka"]["range"]}}


def _close(a, b, tol: float = 1e-9) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return all(_close(a[k], b[k], tol) for k in a if k in b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def reproduce(pub: dict, workers: int, log=print) -> dict:
    """This module's code on the 7 pairs: with the season rule as the published studies had it (no
    reopen / data-break rule), it must give their numbers; with the current rule, the 7-pair figures
    for comparison."""
    t0 = time.time()
    codes = list(PAIRS)
    A7, _, _ = season_frame(codes, None, log=None, current=False)
    old1 = calls_test(A7)["periods"]
    old2 = {p: promote_period(A7[A7["period"] == p]) for p in PERIODS}
    got1 = {p: {k: {q: old1[p][k].get(q) for q in ("n", "hit", "t")} for k in ("all", "high")} for p in PERIODS}
    got2 = {p: {"base": old2[p]["base"]["high"], "new": old2[p]["new"]["high"], "promoted": old2[p]["promoted"]}
            for p in PERIODS}
    cur = live_rule(A7)
    cur["period"] = np.where(cur["origin"] >= SPLIT.value, "test", "tune")
    now1, now2 = calls_test(cur), promote_test(cur)
    if log:
        log(f"reproduce: season done ({time.time() - t0:.0f}s)")
    tr = trade_test(codes, None, holdout=False, log=None, checks=False)
    got3 = {name: {p: {k: x[p].get(k) for k in TRADE_KEYS} for p in PERIODS} for name, x in tr["rules"].items()}
    if log:
        log(f"reproduce: trade done ({time.time() - t0:.0f}s)")
    bres, _ = band_collect(codes, workers, log=None, t0=t0)
    got4, got5 = shape_test(bres), planb_test(bres)
    if log:
        log(f"reproduce: bands done ({time.time() - t0:.0f}s)")
    match = {"calls": _close(got1, pub["calls"]), "promote": _close(got2, pub["promote"]),
             "trade": _close(got3, pub["trade"], 1e-6), "shape": _close(got4, pub["shape"], 1e-7),
             "planb": _close(got5, pub["planb"], 1e-7)}
    return {"match": match, "calls_now": {p: now1["periods"][p] for p in PERIODS}, "calls_now_pass": now1["pass"],
            "promote_now": {p: now2["periods"][p] for p in PERIODS}, "promote_now_pass": now2["pass"],
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ run

def run(log=print, workers: int = WORKERS, with_reproduce: bool = True) -> dict:
    t0 = time.time()
    missing = [c for c in CODES if not (history.HIST_DIR / history.DUKA_DIR / f"{c}_1h.csv").exists()]
    if missing:
        raise FileNotFoundError(f"hold-out bars missing: {missing}")
    res: dict = {"codes": list(CODES), "split": str(SPLIT.date()), "prereg": PREREG,
                 "settings": {"t_min": season.T_MIN, "t_high": season.T_HIGH, "window": season.WINDOW,
                              "promote": PROMOTE, "rule_1h": STUDY_RULES["1h"], "retired": {"carry_mom": trade.RETIRED["carry_mom"]},
                              "rule_1d": STUDY_RULES["1d"], "band_nu": STUDY_BAND_NU,
                              "half_life": {tf: engine.TIMEFRAMES[tf].half_life for tf in ("1h", "1d")},
                              "old_daily_nu": OLD_DAILY_NU, "plan_b": PLAN_B}}
    A, checks, raw = season_frame(CODES, HPAIRS, log)
    res["season_check"] = checks
    res["calendar"] = calendar_check()
    res["calls"] = calls_test(A)
    res["promote"] = promote_test(A)
    res["promote"]["live_check"] = promote_live_check(A, raw)
    res["span"] = {c: [str(raw[c].index[0]), str(raw[c].index[-1]), int(len(raw[c]))] for c in CODES}
    res["rows"] = {p: int((A["period"] == p).sum()) for p in PERIODS}
    res["first_origin"] = str(pd.Timestamp(int(A["origin"].min()), tz="UTC"))
    del A, raw
    log(f"tests 1-2 done ({time.time() - t0:.0f}s)")
    res["trade"] = trade_test(CODES, HPAIRS, holdout=True, log=log)
    log(f"test 3 done ({time.time() - t0:.0f}s)")
    bres, binfo = band_collect(CODES, workers, log, t0)
    res["bands_info"] = binfo
    res["shape"], res["planb"] = shape_test(bres), planb_test(bres)
    res["bands"] = {tf: {h: {"n": x["n"], "periods": x["periods"],
                             "base": {"k_median": x["variants"]["base"]["k_median"],
                                      "replay_check": x["variants"]["base"].get("replay_check")}}
                         for h, x in bres[tf].items()} for tf in bres}
    log(f"tests 4-5 done ({time.time() - t0:.0f}s)")
    res["published"] = published()
    if with_reproduce:
        res["reproduce"] = reproduce(res["published"], workers, log)
        log(f"7-pair reproduction done ({time.time() - t0:.0f}s): {res['reproduce']['match']}")
    res["seconds"] = round(time.time() - t0, 1)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "holdout.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json), encoding="utf-8")
    (REPORT_DIR / "holdout.md").write_text(report(res), encoding="utf-8")
    log(f"done in {res['seconds']} s")
    return res


def _json(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (pd.Timestamp, date)):
        return str(x)
    return str(x)


# ------------------------------------------------------------------ report

def _p(x, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x * 100:.{d}f}%"


def _f(x, d: int = 2, sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _nh(x: dict | None) -> str:
    return f"{x['n']:,} / {_p(x.get('hit'))}" if x and x.get("n") else "0 / —"


def _nht(x: dict | None) -> str:
    return f"{_nh(x)} (t {_f(x.get('t'))})" if x and x.get("n") else "0 / —"


def _ok(b) -> str:
    return "○" if b else "×"


def _mil(x: float | None) -> str:
    if x is None:
        return "—"
    v = x * 1000
    return f"{v:+.2f}" if abs(v) < 1 else f"{v:+.1f}"


def _pv(p) -> str:
    return "—" if p is None else "<0.001" if p < 0.001 else f"{p:.3f}"


def _verdict(x: dict) -> str:
    if x.get("undecided"):
        return "判定不能"
    return "**合格**" if x.get("pass") else "**不合格**"


def _trade_cell(x: dict) -> str:
    if not x.get("n"):
        return "0 回"
    return f"{x['n']:,} 回・勝率 {x['win']:.0%}・{_f(x['pips'], 1)} pips (t {_f(x.get('t'))})"


def _summary(res: dict) -> list[str]:
    pub, rep = res["published"], res.get("reproduce", {})
    c, pr, tr, sh, pb = res["calls"], res["promote"], res["trade"], res["shape"], res["planb"]
    L = ["## 結論", "",
         "| 検定 | 合否 | ホールドアウト6ペア: 2006〜2016 / 2017〜 | 元の7ペア: 既存の報告 (調整 〜2016 / 検証 2017〜) | 本番のルール |",
         "|---|---|---|---|---|"]
    cp, pp = c["periods"], pub["calls"]
    cell = " / ".join(f"方向 {_nht(cp[p]['all'])}、高 {_nh(cp[p]['high'])}" for p in PERIODS)
    pcell = " / ".join(f"方向 {_nh(pp[p]['all'])} (t {_f(pp[p]['all'].get('t'))})、高 {_nh(pp[p]['high'])}" for p in PERIODS)
    L.append(f"| 1. 時間帯の偏り | {_verdict(c)} | {cell} | {pcell} | {'そのまま' if c['pass'] else '見直しを検討'} |")
    P, Q = pr["periods"], pub["promote"]
    cell = " / ".join(f"高 {_p(P[p]['base']['high'].get('hit'))} → {_p(P[p]['new']['high'].get('hit'))}、格上げ {_nh(P[p]['promoted'])}"
                      for p in PERIODS)
    pcell = " / ".join(f"高 {_p(Q[p]['base'].get('hit'))} → {_p(Q[p]['new'].get('hit'))}、格上げ {_nh(Q[p]['promoted'])}"
                       for p in PERIODS)
    L.append(f"| 2. ロールオーバーの格上げ | {_verdict(pr)} | {cell} | {pcell} | {'そのまま' if pr['pass'] else '見直しを検討'} |")
    a, b = tr["rules"]["carry_mom_vol"], tr["rules"]["carry_mom"]
    cell = " / ".join(f"{_f(a[p].get('pips'), 1)} 対 {_f(b[p].get('pips'), 1)} pips" for p in PERIODS)
    T = pub["trade"]
    pcell = " / ".join(f"{_f(T['carry_mom_vol'][p]['pips'], 1)} 対 {_f(T['carry_mom'][p]['pips'], 1)} pips" for p in PERIODS)
    both_neg = all((x[p].get("pips") or 0) < 0 for x in (a, b) for p in ("test",))
    L.append(f"| 3. 1時間足の売買ルール (carry_mom_vol 対 carry_mom、1回平均) | {_verdict(tr)} | {cell} | {pcell} | "
             f"{'そのまま' if tr['pass'] else '見直しを検討' + (' (2017〜 はどちらもマイナス)' if both_neg else '')} |")
    cell = " / ".join("・".join(f"{h}日 {_mil(sh[h][p]['ls_diff'])}" for h in ("1", "20")) for p in PERIODS)
    pcell = " / ".join("・".join(f"{h}日 {_mil(pub['shape'][h][p]['ls_diff'])}" for h in ("1", "20")) for p in PERIODS)
    ok_h = {h: all(sh[h][p]["pass"] for p in PERIODS) for h in ("1", "20")}
    todo = ("そのまま" if sh["pass"] else "、".join(f"{h}営業日先は{'そのまま' if ok_h[h] else '見直しを検討'}" for h in ("20", "1")))
    L.append(f"| 4. 日足のレンジの形 (対数スコアの差 ×1000) | {_verdict(sh)} | {cell} | {pcell} | {todo} |")
    keys = [f"{tf}_{h}" for tf, hs in PLAN_B_CHANGED.items() for h in hs]
    lab = {k: (f"{k.split('_')[1]}時間" if k.startswith("1h") else f"{k.split('_')[1]}日") for k in keys}
    cell = " / ".join("・".join(f"{lab[k]} {_mil(pb[k][p]['ls_diff'])}{'' if pb[k][p]['cov_ok'] else '(ずれ悪化)'}" for k in keys)
                      for p in PERIODS)
    pcell = " / ".join("・".join(f"{lab[k]} {_mil(pub['planb'][k][p]['ls_diff'])}{'' if pub['planb'][k][p]['cov_ok'] else '(ずれ悪化)'}"
                                for k in keys) for p in PERIODS)
    L.append(f"| 5. 案B (対数スコアの差 ×1000、本番の設定との比較) | {_verdict(pb)} | {cell} | {pcell} | "
             f"{'採用の条件を満たす' if pb['pass'] else '採用しない (今のまま)'} |")
    L.append("")
    L.append("7ペアの「既存の報告」は research/rollover.md (Dukascopy 2004〜)、research/signals.md、research/bands.md の数字で、"
             "1・2 は時間帯の偏りの以前の規則 (市場の再開直後の足とデータの途切れを除く前) のものです。5. の7ペアの数字は、bands.json の"
             "同じ計算結果から本番の設定 (案A 採用後) との差を求め直したもの (bands.md の表は以前の設定との差) です。")
    if rep:
        n1, n2 = rep["calls_now"], rep["promote_now"]
        L.append("同じコードで今の規則のまま7ペアを計算し直すと、1. は " +
                 " / ".join(f"方向 {_nht(n1[p]['all'])}、高 {_nh(n1[p]['high'])}" for p in PERIODS) +
                 "、2. は " + " / ".join(f"高 {_p(n2[p]['base']['high'].get('hit'))} → {_p(n2[p]['new']['high'].get('hit'))}、"
                                        f"格上げ {_nh(n2[p]['promoted'])}" for p in PERIODS) + " です (前半は2004〜2016)。")
    L.append("")
    return L


def _bullets(res: dict) -> list[str]:
    c, pr, tr, sh, pb = res["calls"], res["promote"], res["trade"], res["shape"], res["planb"]
    L = []
    te, tu = c["periods"]["test"], c["periods"]["tune"]
    cond = c["conditions"]
    L.append(f"- **1. 時間帯の偏り: {_verdict(c).strip('*')}。** 2017〜 は方向を示した回 {_nh(te['all'])} (t {_f(te['all'].get('t'))})、"
             f"確度: 高 {_nh(te['high'])} (条件: 50%超 {_ok(cond['hit'])}・t ≥ 2 {_ok(cond['t'])}・高が全体より上 {_ok(cond['high'])})。"
             f"2006〜2016 は {_nh(tu['all'])} (t {_f(tu['all'].get('t'))})、高 {_nh(tu['high'])}。")
    P = pr["periods"]
    parts = []
    for p in PERIODS:
        x = P[p]
        parts.append(f"{PERIOD_JA[p]}: 今の確度: 高 {_nh(x['base']['high'])} → 格上げ後 {_nh(x['new']['high'])}、"
                     f"格上げされた足 {_nh(x['promoted'])} (条件: 格上げの的中率 {_ok(x['conditions']['promoted'])}・"
                     f"高の的中率 {_ok(x['conditions']['high'])})")
    L.append(f"- **2. ロールオーバーの格上げ: {_verdict(pr).strip('*')}。** " + "。".join(parts) + "。")
    a, b = tr["rules"]["carry_mom_vol"], tr["rules"]["carry_mom"]
    L.append(f"- **3. 1時間足の売買ルール: {_verdict(tr).strip('*')}。** carry_mom_vol は " +
             " / ".join(f"{PERIOD_JA[p]} {_trade_cell(a[p])}" for p in PERIODS) + "、carry_mom は " +
             " / ".join(f"{PERIOD_JA[p]} {_trade_cell(b[p])}" for p in PERIODS) +
             f" (条件: 前半 {_ok(tr['conditions']['tune'])}・後半 {_ok(tr['conditions']['test'])})。")
    L.append(f"- **4. 日足のレンジの形: {_verdict(sh).strip('*')}。** 対数スコアの差 (×1000、新 − 旧) は 1営業日先 (ν 10 → 6) "
             + " / ".join(f"{PERIOD_JA[p]} {_mil(sh['1'][p]['ls_diff'])}" for p in PERIODS) + "、20営業日先 (ν 30 → 5) "
             + " / ".join(f"{PERIOD_JA[p]} {_mil(sh['20'][p]['ls_diff'])}" for p in PERIODS) + "。")
    parts = []
    for tf, hs in PLAN_B_CHANGED.items():
        for h in hs:
            x = pb[f"{tf}_{h}"]
            lab = f"{h}時間先" if tf == "1h" else f"{h}営業日先"
            parts.append(f"{lab} " + " / ".join(f"{_mil(x[p]['ls_diff'])} (ずれ {_pp(x[p]['cov_diff'])})" for p in PERIODS)
                         + f" {_ok(x['pass'])}")
    L.append(f"- **5. 案B: {_verdict(pb).strip('*')}。** 本番の設定との対数スコアの差 (×1000) と的中率の名目からのずれの変化 "
             "(ポイント、マイナスが改善) は、" + "、".join(parts) + " (それぞれ 2006〜2016 / 2017〜)。")
    return L


def _notes(res: dict) -> list[str]:
    """How to read the results (written after they were seen; no verdict changes)."""
    c, pr, tr, sh, pb = res["calls"], res["promote"], res["trade"], res["shape"], res["planb"]
    rep_ = res.get("reproduce", {})
    L = ["### 結果の読み方 (結果を見た後に加えた注記。合否は変えていません)", ""]
    rb_ = c["roll_bar"]["test"]
    now = rep_.get("calls_now", {}).get("test")
    few = sorted(((code, x["high"]["n"]) for code, x in c["pairs"]["test"].items()), key=lambda v: v[1])[:2]
    L.append(f"- **1.** 偏りはホールドアウトでも本物ですが、元の7ペアより弱めです (2017〜 の確度: 高 {_p(c['periods']['test']['high'].get('hit'))}"
             + (f"、7ペアを同じ規則で {_p(now['high'].get('hit'))}" if now else "") + ")。2017〜 はロールオーバーの足が "
             f"{_nh(rb_['roll']['all'])}、その他の足が {_nh(rb_['other']['all'])} で、効果の中心はロールオーバーです。確度: 高 の回数は"
             "金利差しだいで、2017〜 は " + "・".join(f"{c_} {n:,}回" for c_, n in few) + " と少ないペアもあります。")
    P = pr["periods"]
    top = {p: sorted(((code, x[p]["promoted"]["n"]) for code, x in pr["pairs"].items() if x[p]["promoted"]["n"]),
                     key=lambda v: -v[1])[:3] for p in PERIODS}
    L.append("- **2.** 格上げされた足は " + " / ".join(f"{PERIOD_JA[p]} {_nh(P[p]['promoted'])}" for p in PERIODS)
             + " で、どちらの期間も今の確度: 高 の的中率を上回りました。ただし格上げは金利差の大きいペア・時期に集中します ("
             + " / ".join(f"{PERIOD_JA[p]}: " + "・".join(f"{c_} {n:,}" for c_, n in top[p]) for p in PERIODS)
             + ")。2017〜 の確度: 高 の的中率の上がり幅は "
             + f"{(P['test']['new']['high']['hit'] - P['test']['base']['high']['hit']) * 100:+.1f} ポイントと小さく、効果は「確度: 高 を"
             "的中率を下げずに増やせる」程度です。"
             + ("USDCAD は格上げが0回だったので、T+1 で数えても結果は同じでした。"
                if all(pr["pairs"].get("USDCAD", {}).get(p, {}).get("promoted", {}).get("n", 0) == 0 for p in PERIODS) else ""))
    a, b = tr["rules"]["carry_mom_vol"], tr["rules"]["carry_mom"]
    if not tr["pass"]:
        L.append(f"- **3.** 2006〜2016 は carry_mom_vol が上回りました ({_f(a['tune']['pips'], 1)} 対 {_f(b['tune']['pips'], 1)} pips) が、"
                 f"2017〜 は {_f(a['test']['pips'], 1)} 対 {_f(b['test']['pips'], 1)} pips と差がなく、**どちらのルールも損失** です "
                 f"(t {_f(a['test'].get('t'))} / {_f(b['test'].get('t'))})。本番相当のコストだけで数えても {_f(a['test'].get('pips_live'), 1)} / "
                 f"{_f(b['test'].get('pips_live'), 1)} pips なので、スプレッドの広さだけが理由ではありません。7ペアで見つけた「金利差 + 5日間の流れ」"
                 "の優位は、ほかのペアの2017年以降には広がっていません。見送りと最長120本への変更が以前のルールより良いという根拠も、"
                 "ホールドアウトの 2017〜 では得られませんでした。本番の1時間足のシグナルは優位の確かめられていない参考表示で、"
                 "表示を続けるかどうかを含めて見直すべきです。日足の金利差ルール (参考) も 2017〜 は "
                 f"{_f(tr['rules']['carry_1d']['test'].get('pips'), 1)} pips (t {_f(tr['rules']['carry_1d']['test'].get('t'))}) でした。")
    bad = [h for h in ("1", "20") if not all(sh[h][p]["pass"] for p in PERIODS)]
    if bad:
        x = sh["1"]
        L.append(f"- **4.** 20営業日先 (ν 30 → 5) はホールドアウトでも両方の期間で良くなりました ({_mil(sh['20']['tune']['ls_diff'])} / "
                 f"{_mil(sh['20']['test']['ls_diff'])})。1営業日先 (ν 10 → 6) は" +
                 ("ホールドアウトで逆に少し悪くなり" if "1" in bad else "") +
                 f" ({_mil(x['tune']['ls_diff'])} / {_mil(x['test']['ls_diff'])}、p {_pv(x['tune']['p'])} / {_pv(x['test']['p'])})、95%レンジの的中が "
                 f"{_p(x['tune']['cover_old']['95'])} → {_p(x['tune']['cover_new']['95'])}、{_p(x['test']['cover_old']['95'])} → "
                 f"{_p(x['test']['cover_new']['95'])} と名目を超えました。7ペアでの改善 ({_mil(res['published']['shape']['1']['tune']['ls_diff'])} / "
                 f"{_mil(res['published']['shape']['1']['test']['ls_diff'])}) も小さく、1営業日先の変更は"
                 "どちらの向きにも確かな差がありません。「確かな根拠のある変更」という基準では、1営業日先は以前の ν = 10 に戻すのが筋です。")
    if not pb["pass"]:
        fails = [k for k, x in pb.items() if isinstance(x, dict) and not x["pass"]]
        why = []
        for k in fails:
            x = pb[k]
            lab = f"{k.split('_')[1]}時間先" if k.startswith("1h") else f"{k.split('_')[1]}営業日先"
            why.append(f"{lab} (対数スコア {_mil(x['tune']['ls_diff'])} / {_mil(x['test']['ls_diff'])}、的中率のずれ "
                       f"{_p(x['tune']['cov_ref'], 2)} → {_p(x['tune']['cov_new'], 2)} / {_p(x['test']['cov_ref'], 2)} → {_p(x['test']['cov_new'], 2)})")
        rest = [k for k, x in pb.items() if isinstance(x, dict) and x["pass"]]
        L.append("- **5.** 案B は " + "、".join(why) + " で条件を満たさず、事前に決めた基準では不合格です。対数スコアはどの予測先・どちらの期間でも"
                 "上がり、" + "・".join((f"{k.split('_')[1]}時間先" if k.startswith("1h") else f"{k.split('_')[1]}営業日先") for k in rest)
                 + f" はすべての条件を満たしました。不合格の理由になったずれの悪化は最大 "
                 f"{max(pb[k][p]['cov_diff'] for k in fails for p in PERIODS) * 100:.3f} ポイントと実質的には変化なしですが、基準は変えません。"
                 "「4時間先を除いた案B」はホールドアウトで条件を満たしますが、これは結果を見てから選んだ形なので、この確認の対象外です"
                 " (採用するなら、台帳の実績など別の期間で確かめ直す必要があります)。")
    L.append("")
    return L


def _decisions(res: dict) -> list[str]:
    """What the live system took from the results (changes made after this study)."""
    c, tr, sh = res["calls"], res["trade"], res["shape"]
    L = ["### 本番への反映 (この確認の後に行った変更)", "",
         "基準は「設計に使っていないデータでも確かめられたものだけを本番に残す」です。", ""]
    L.append(("- **1・2 (時間帯の偏りとロールオーバーの格上げ):** " + ("合格したので本番のまま残しました。"
              if c["pass"] and res["promote"]["pass"] else "不合格のものは本番から外しました。")
              + "予想ローソク足の「向きに根拠のある足」「確度が高い足」はこの2つだけを使っています。"))
    t = {k: tr["rules"][k]["test"] for k in ("carry_mom_vol", "carry_1d")}
    L.append("- **3 (売買ルール):** 本番の1時間足の carry_mom_vol と日足の carry は、ホールドアウトの 2017〜 でどちらも損失 "
             f"({t['carry_mom_vol'].get('pips'):+.1f} pips、{t['carry_1d'].get('pips'):+.1f} pips / 1回、コスト・スワップ込み) で、"
             "以前のルールより良いという根拠もありませんでした。根拠のない売買シグナルは出さないことにし、売買の方向 (買い・売り) の表示を"
             "やめました (trade.RULES を空にし、2つのルールは trade.TESTED に定義だけを残しています)。損切り・利確の目安 (ATR から、買い・"
             "売りの両方) は、方向の予想ではない参考の値として表示を続けます。台帳に記録済みのシグナルの決済条件は trade.RETIRED に残し、"
             "検証 (verify) は以前の記録もそのまま確かめられます。")
    one = sh["1"]
    L.append(f"- **4 (日足のレンジの形):** 1営業日先は ν {one['new']} → {one['old']} に戻しました (ホールドアウトで対数スコアが下がり、"
             "7ペアでの改善も小さいため)。20営業日先 (ν 5) は両方で良くなったのでそのままです。")
    L.append("- **5 (案B):** 事前の基準で不合格なので採用しません (今のまま)。")
    L += ["", "時間帯の偏りの的中率 (2017〜、方向を示した回 "
          f"{_p(c['periods']['test']['all'].get('hit'))}・確度: 高 {_p(c['periods']['test']['high'].get('hit'))}) は、"
          "設計に使っていない6ペアで確かめた、この予想の方向の根拠の大きさです。", ""]
    return L


def _pp(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.2f}"


def _calls_section(res: dict) -> list[str]:
    c = res["calls"]
    L = ["## 1. 時間帯の偏り (season.py の方向)", "",
         "ロールオーバー前後を含むすべての時間の、次の1時間の方向です (市場が開いている足、6,000本の履歴がそろった起点から)。",
         "", "| 期間 | 起点 | 方向を示した回: 回数 / 的中率 (t) | 足に占める割合 | 平均 (bp) | 確度: 高: 回数 / 的中率 (t) | 高の95%下限 |",
         "|---|---|---|---|---|---|---|"]
    for p in PERIODS:
        x = c["periods"][p]
        L.append(f"| {PERIOD_JA[p]} | {res['rows'][p]:,} | {_nht(x['all'])} | {_p(x['all'].get('share'), 2)} | "
                 f"{_f(x['all'].get('bp'))} | {_nht(x['high'])} | {_p(x['high'].get('hit_lo'))} |")
    pub, rep_ = res["published"]["calls"], res.get("reproduce", {}).get("calls_now")
    L += ["", "元の7ペア (Dukascopy、前半は2004〜2016):", "",
          "| データ | 期間 | 方向を示した回: 回数 / 的中率 (t) | 確度: 高: 回数 / 的中率 (t) |", "|---|---|---|---|"]
    for p in PERIODS:
        L.append(f"| rollover.md (以前の規則) | {'〜2016' if p == 'tune' else '2017〜'} | {_nht(pub[p]['all'])} | {_nht(pub[p]['high'])} |")
    if rep_:
        for p in PERIODS:
            L.append(f"| 同じコード・今の規則 | {'〜2016' if p == 'tune' else '2017〜'} | {_nht(rep_[p]['all'])} | {_nht(rep_[p]['high'])} |")
    L += ["", "ロールオーバーの足 (ニューヨーク時間17時に始まる月〜木の足) とその他の足:", "",
          "| 期間 | 足 | 方向を示した回 | 確度: 高 |", "|---|---|---|---|"]
    for p in PERIODS:
        for k, lab in (("roll", "ロールオーバーの足"), ("other", "その他")):
            x = c["roll_bar"][p].get(k)
            if x:
                L.append(f"| {PERIOD_JA[p]} | {lab} | {_nht(x['all'])} | {_nh(x['high'])} |")
    L += ["", "ペア別 (回数 / 的中率):", "", "| ペア | 2006〜2016: 方向 | 高 | 2017〜: 方向 | 高 |", "|---|---|---|---|---|"]
    for code in CODES:
        cells = []
        for p in PERIODS:
            x = c["pairs"][p].get(code)
            cells += [_nh(x["all"]) if x else "—", _nh(x["high"]) if x else "—"]
        L.append(f"| {code} | " + " | ".join(cells) + " |")
    L += ["", "年ごと (回数 / 的中率):", "", "| 年 | 方向を示した回 | 確度: 高 |", "|---|---|---|"]
    for y, x in c["years"].items():
        L.append(f"| {y} | {_nh(x['all'])} | {_nh(x['high'])} |")
    L.append("")
    return L


def _promote_rows(label: str, per: dict, seven: bool = False) -> list[str]:
    """Rows of test 2: the high tier before, the promoted calls, the high tier after, and the two conditions.
    ``per[p]`` is promote_period's result, or {"base", "new", "promoted"} of high-tier scores (the 7 pairs)."""
    L = []
    for p in PERIODS:
        x = per[p]
        b, n, up = ((x["base"]["high"], x["new"]["high"], x["promoted"]) if "conditions" in x
                    else (x["base"], x["new"], x["promoted"]))
        ok_p = bool(up.get("n") and up.get("hit") is not None and up["hit"] >= b["hit"] - HIT_TOL - EPS)
        ok_h = bool(up.get("n") and n.get("hit") is not None and n["hit"] >= b["hit"] - HIT_TOL - EPS)
        per_lab = ("〜2016" if p == "tune" else "2017〜") if seven else PERIOD_JA[p]
        L.append(f"| {label} | {per_lab} | {_nh(b)} | {_nh(up)} ({_p(up.get('hit_lo'))}) | "
                 f"{_nh(n)} | {_ok(ok_p)} {_ok(ok_h)} |")
    return L


def _promote_section(res: dict) -> list[str]:
    pr = res["promote"]
    L = ["## 2. ロールオーバーの格上げ", "",
         "e は予測の起点の日に分かっていた金利の差 (history.rates_panel、本番の rates.rate_diff と同じ系列・同じ遅れ) と、祝日の暦で数えたスポット日の"
         "移動日数から計算しました。条件の列は「格上げの的中率 ≥ 今の高 − 1ポイント」「格上げ後の高 ≥ 今の高 − 1ポイント」です。", "",
         "| データ | 期間 | 今の確度: 高 (\\|t\\| ≥ 4) | 格上げされた足 (95%下限) | 格上げ後の確度: 高 | 条件 |", "|---|---|---|---|---|---|"]
    L += _promote_rows("6ペア (本番の規則、T+2)", pr["periods"])
    if "usdcad_t1" in pr:
        L += _promote_rows("補助: USDCAD を T+1 で", pr["usdcad_t1"])
    L += _promote_rows("補助: USDCAD を除く5ペア", pr["without_usdcad"])
    L += _promote_rows("7ペア: rollover.md (以前の規則)", res["published"]["promote"], True)
    if res.get("reproduce"):
        L += _promote_rows("7ペア: 同じコード・今の規則", res["reproduce"]["promote_now"], True)
    L += ["", "ペア別 (回数 / 的中率):", "", "| ペア | 2006〜2016: 今の高 | 格上げ | 2017〜: 今の高 | 格上げ |", "|---|---|---|---|---|"]
    for code in CODES:
        x = pr["pairs"].get(code, {})
        cells = []
        for p in PERIODS:
            y = x.get(p, {})
            cells += [_nh(y.get("high")), _nh(y.get("promoted"))]
        L.append(f"| {code} | " + " | ".join(cells) + " |")
    rb_ = pr["roll_bars"]
    L += ["", "ロールオーバーの足 (月〜木の17時): " + "、".join(
        f"{PERIOD_JA[p]} {rb_[p]['n']:,} 本 (暦で日数が単純な数え方と違う足 {rb_[p]['cal_differs']:,} 本、|e| ≥ 2 bp の足 {rb_[p]['e_ge_2']:,} 本)"
        for p in PERIODS) + "。", ""]
    return L


def _trade_section(res: dict) -> list[str]:
    tr = res["trade"]
    L = ["## 3. 1時間足の売買ルール", "",
         "research_signals.run_variant (research_trade.simulate: 足の終値でエントリー、1本の中で損切りと利確の両方に届いたら損切り、1ペアに"
         "同時に1つ) で計算しました。本番相当のコスト (往復、pips) はペアの記録されたスプレッドの中央値です: "
         + "、".join(f"{c} {v:.2f}" for c, v in tr["live_cost"].items()) + "。記録されたスプレッドの中央値 (2006〜2016 / 2017〜): "
         + "、".join(f"{c} {v['tune']:.2f} / {v['test']:.2f}" for c, v in tr["spread_median"].items()) + "。", "",
         "| ルール | データ・期間 | 取引数 | 勝率 | 平均pips | 合計pips | t (月ごとの R) | 本番相当のコストだけ: 平均pips | 1回の平均: コスト / スワップ |",
         "|---|---|---|---|---|---|---|---|---|"]
    names = {"carry_mom_vol": "carry_mom_vol (本番の1時間足)", "carry_mom": "carry_mom (以前の1時間足)",
             "carry_1d": "carry (本番の日足、参考)"}
    for name, lab in names.items():
        for src, per, seven in (("6ペア", tr["rules"][name], False), ("7ペア (signals.md)", res["published"]["trade"][name], True)):
            for p in PERIODS:
                x = per[p]
                pl = ("〜2016" if p == "tune" else "2017〜") if seven else PERIOD_JA[p]
                if not x.get("n"):
                    L.append(f"| {lab} | {src} {pl} | 0 | — | — | — | — | — | — |")
                    continue
                L.append(f"| {lab} | {src} {pl} | {x['n']:,} | {x['win']:.0%} | {_f(x['pips'], 1)} | {_f(x['total_pips'], 0)} | "
                         f"{_f(x.get('t'))} | {_f(x.get('pips_live'), 1)} | {_f(x.get('cost'), 2, False)} / {_f(x.get('swap'), 2)} |")
    L += ["", "ペア別の1回平均pips (取引数):", "",
          "| ペア | 2006〜2016: carry_mom_vol | carry_mom | 2017〜: carry_mom_vol | carry_mom | 日足 2006〜2016 | 日足 2017〜 |",
          "|---|---|---|---|---|---|---|"]
    for code in CODES:
        cells = []
        for p in PERIODS:
            for name in ("carry_mom_vol", "carry_mom"):
                x = tr["rules"][name]["pair"].get(code, {}).get(p, {})
                cells.append(f"{_f(x['pips'], 1)} ({x['n']:,})" if x.get("n") else "—")
        for p in PERIODS:
            x = tr["rules"]["carry_1d"]["pair"].get(code, {}).get(p, {})
            cells.append(f"{_f(x['pips'], 1)} ({x['n']:,})" if x.get("n") else "—")
        L.append(f"| {code} | " + " | ".join(cells) + " |")
    L += ["", "3年ごと (エントリーの年、取引数 / 平均pips / t):", "", "| 期間 | carry_mom_vol | carry_mom |", "|---|---|---|"]
    blocks = sorted(set(tr["rules"]["carry_mom_vol"]["block"]) | set(tr["rules"]["carry_mom"]["block"]),
                    key=lambda b: (b != "〜2005", b))
    for blk in blocks:
        cells = []
        for name in ("carry_mom_vol", "carry_mom"):
            x = tr["rules"][name]["block"].get(blk)
            cells.append(f"{x['n']:,} / {_f(x['pips'], 1)} / {_f(x.get('t'))}" if x and x.get("n") else "—")
        L.append(f"| {blk} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def _cov(c: dict) -> str:
    return " / ".join(_p(c[lv]) for lv in rb.LEVELS)


def _shape_section(res: dict) -> list[str]:
    sh, pub = res["shape"], res["published"]["shape"]
    L = ["## 4. 日足のレンジの形 (1営業日先 ν 10 → 6、20営業日先 ν 30 → 5)", "",
         "research_bands の日足の計算 (Dukascopy の1時間足から作ったロンドン日足、本番の生の σ、k の学習の再現。半減期 120 件、6ペアまとめて) です。"
         "k は ν によらないので、80%レンジは同じです。p は Diebold-Mariano 検定 (日ごとの平均、予測期間の重なりを考慮)。", "",
         "| 予測先 | データ | 期間 | 件数 | 50/80/95%レンジ: 旧 ν | 新 ν | 対数スコアの差 (×1000) | p |", "|---|---|---|---|---|---|---|---|"]
    for h in ("1", "20"):
        for lab, src in (("6ペア", sh), ("7ペア (bands.json)", pub)):
            for p in PERIODS:
                x = src[h][p]
                L.append(f"| {h}営業日先 (ν {src[h]['old']} → {src[h]['new']}) | {lab} | {PERIOD_JA[p] if lab == '6ペア' else ('〜2016' if p == 'tune' else '2017〜')} | "
                         f"{x['n']:,} | {_cov(x['cover_old'])} | {_cov(x['cover_new'])} | {_mil(x['ls_diff'])} | {_pv(x['p'])} |")
    L.append("")
    return L


def _planb_section(res: dict) -> list[str]:
    pb, pub = res["planb"], res["published"]["planb"]
    L = ["## 5. 案B (k の半減期と ν を予測先ごとに)", "",
         "本番の設定 (1時間足: 半減期 400 件、ν 5・5・6、日足: 半減期 120 件、ν 6・10・15・5) と案B を、変わる予測先ごとに比べました。"
         "対数スコアは幅 (生の σ × k) の違いを含む値動きそのものの対数スコア、「ずれ」は 50/80/95%レンジの的中率と名目の差の平均です。"
         "半減期は6ペアまとめた判定済みの予測の件数で数えるので、同じ半減期でも7ペアより約 7/6 倍長い期間になります。", "",
         "| 予測先 | 変更 | データ | 期間 | 件数 | k の中央値 | 50/80/95%レンジ: 本番 | 案B | ずれ: 本番 → 案B | 対数スコアの差 (×1000) | 条件 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key in pb:
        if key == "pass":
            continue
        tf, h = key.split("_")
        lab = f"{h}時間先" if tf == "1h" else f"{h}営業日先"
        for src_lab, src in (("6ペア", pb), ("7ペア (bands.json)", pub)):
            x = src[key]
            chg = f"半減期 {x['half_life'][0]:g} → {x['half_life'][1]:g}、ν {x['nu'][0]} → {x['nu'][1]}"
            for p in PERIODS:
                y = x[p]
                per = PERIOD_JA[p] if src_lab == "6ペア" else ("〜2016" if p == "tune" else "2017〜")
                L.append(f"| {lab} | {chg} | {src_lab} | {per} | {y['n']:,} | {y['k_ref']:.3f} → {y['k_new']:.3f} | "
                         f"{_cov(y['cover_ref'])} | {_cov(y['cover_new'])} | {_p(y['cov_ref'], 2)} → {_p(y['cov_new'], 2)} | "
                         f"{_mil(y['ls_diff'])} | {_ok(y['ls_ok'])} {_ok(y['cov_ok'])} |")
    L += ["", "条件の列は「対数スコアが上がる」「ずれが大きくならない」です。", ""]
    return L


def _planb_code() -> list[str]:
    return ["### 案B を採用する場合の変更", "",
            "engine.py (ν と予測先ごとの半減期):", "", "```python",
            'BAND_NU = {"15m": {1: 5, 4: 5, 16: 4}, "1h": {1: 5, 4: 5, 24: 8}, "1d": {1: 6, 5: 10, 10: 10, 20: 6}}',
            "# k half-life per horizon (scored forecasts, pairs pooled); horizons not listed use Timeframe.half_life",
            'HALF_LIFE = {"1h": {1: 400.0, 4: 1600.0, 24: 1600.0}, "1d": {1: 120.0, 5: 120.0, 10: 480.0, 20: 480.0}}',
            "", "", "def half_life(tf: Timeframe, h: int) -> float:",
            "    return HALF_LIFE.get(tf.key, {}).get(h, tf.half_life)", "```", "",
            "learning.py (`learn` が予測先ごとの半減期を受け取る):", "", "```python",
            "def learn(prior_rec: dict | None, samples: dict[int, list[dict]], horizons, half_life) -> dict[int, HorizonState]:",
            '    """``half_life``: one value for every horizon, or a callable h -> half-life."""',
            "    out = {}", "    for h in horizons:", '        prior = prior_rec["h"].get(str(h)) if prior_rec else None',
            "        hl = half_life(h) if callable(half_life) else half_life",
            "        out[h] = learn_horizon(prior, samples.get(h, []), hl)", "    return out", "```", "",
            "呼び出し側 (pipeline.py・audit.py・api._learning の `learn(..., tf.half_life)` を "
            "`learn(..., lambda h: engine.half_life(tf, h))` に、backtest._replay の `learn_arrays(..., tf.half_life, ...)` を "
            "`learn_arrays(..., engine.half_life(tf, int(hkey)), ...)` に)。api.py の models.json の half_life も予測先ごとに出します。"
            "engine.py と learning.py が変わるので model_version が変わり、抜き取りの再計算はこれまでどおり同じ版の予測だけが対象です。", ""]


def _method_section(res: dict) -> list[str]:
    cal = res["calendar"]
    L = ["## データと方法", "",
         "- **データ:** " + "、".join(f"{c} {v[0][:10]}〜{v[1][:10]} ({v[2]:,} 本)" for c, v in res["span"].items()) +
         f"。1・2 の最初の起点は {res['first_origin'][:10]} です。",
         "- **時間帯の偏り (1・2):** research_rollover.build (季節の枠の累積和で全時点をまとめて計算) を、市場の再開直後の足に方向を示さず、"
         "起点の直後に続かない予測を除いて使いました (今の season.bar_drift と research_direction.session_eval の規則)。",
         "- **売買 (3):** research_signals.prepare / run_variant / split / breakdown をそのまま使い、ペアの情報 (pip、通貨) と本番相当のコストだけを"
         "ホールドアウトのペアのものにしました。",
         "- **レンジ (4・5):** research_bands.hourly_pair / daily_pair / analyse_h をそのまま使い、Dukascopy だけを計算しました (Yahoo のデータは"
         "ホールドアウトのペアにありません)。案B の半減期 ×4 は research_bands の事前に決めた候補「k の半減期 ×4」と同じ計算です。",
         "", "### 祝日の暦 (CHF・CAD・NZD、このモジュールで規則から計算)", "",
         "- **CHF (チューリッヒ、SIX Interbank Clearing の休業日):** 1月1日・2日、聖金曜日、復活祭の月曜日、キリスト昇天祭 (復活祭の39日後)、"
         "聖霊降臨祭の月曜日 (50日後)、5月1日、8月1日、12月25日・26日。土日に重なっても振替はありません。限界: 12月24日・31日 (取引所は休みでも"
         "決済は動く日) とチューリッヒの半日の休み (Sechseläuten など) は含めていません。",
         "- **CAD (トロント、QuantLib の Canada Settlement と同じ規則):** 1月1日・7月1日・11月11日 (土日なら次の月曜)、家族の日 (2月の第3月曜、"
         "2008年から)、聖金曜日、ビクトリアデー (5月24日以前の最後の月曜)、8月の第1月曜、9月の第1月曜、真実と和解の日 (9月30日、2021年から、"
         "土日なら次の月曜)、10月の第2月曜、12月25日・26日 (土日なら次の空いている平日)。限界: 家族の日はオンタリオ州の休日で、カナダの決済"
         "システム (Lynx) は開いている可能性があり、真実と和解の日を決済の休日とした年も確かめていません。**USD/CAD のスポットは市場の慣行では"
         "T+1** で、本番の spot_date (T+2) で数えると週末をまたぐ3日分のずれが木曜日ではなく水曜日に来ます (合否は本番の規則で判定し、T+1 は補助の確認)。",
         "- **NZD (ウェリントンとオークランド):** 1月1日・2日と12月25日・26日 (土日なら次の空いている平日)、ウェリントン記念日 (1月22日に一番近い"
         "月曜)、オークランド記念日 (1月29日に一番近い月曜)、ワイタンギ・デー (2月6日) とアンザック・デー (4月25日) (2014年から土日なら次の月曜)、"
         "聖金曜日、復活祭の月曜日、6月の第1月曜、10月の第4月曜、マタリキ (2022年から法律の日付)、2022年9月26日 (エリザベス女王の追悼の日)。"
         "限界: 決済の休日は両都市の休日の和としました (NZD の決済の慣行)。他の地方の記念日は含めていません。",
         "- スポット日は fxcalendar.spot_date (T+2: 1営業日目はドル以外の通貨の営業日、受け渡し日は両通貨とドルの営業日)、休日は2006〜2026年に"
         "一度だけ計算した暦から。ロールオーバーの日数の分布 (月〜木の各営業日、2006〜2026年):", "",
         "| ペア | ロールオーバー | 単純な数え方と違う日 | 日数の内訳 | fxcalendar.roll_days との不一致 |", "|---|---|---|---|---|"]
    for code, x in cal.items():
        L.append(f"| {code} | {x['rolls']:,} | {x['differ']:,} | " + "、".join(f"{k}日 {v:,}" for k, v in x["counts"].items()) +
                 f" | {x.get('vs_fxcalendar_mismatch', '— (暦なし)')} |")
    L.append("")
    return L


def _check_section(res: dict) -> list[str]:
    sc = res["season_check"]
    lc = res["promote"]["live_check"]
    sig = res["trade"].get("signal_check", {})
    bi = res["bands_info"]["checks"]
    L = ["## 確認", "",
         f"- **時間帯の偏り:** 各ペアで無作為に選んだ日の起点 (計 {sum(x['origins'] for x in sc.values()):,}) を season.slot_stats / bar_drift で"
         f"計算し直すと、t の差は最大 {max(x['max_abs_t'] for x in sc.values()):.1e}、方向の食い違いは {sum(x['mismatch'] for x in sc.values())} 件でした。",
         f"- **格上げ:** 格上げされたすべての足と、その他のロールオーバーの足の抜き取り ({lc['origins']:,} 起点、うち格上げ {lc['raised']:,}) を "
         f"season.slot_stats / bar_drift / promote と上の暦の roll_shift で計算し直すと、確度: 高 の判定の食い違いは {lc['mismatch']} 件 "
         f"(取引時間の暦の次の足と保存された足が違う {lc['skipped']} 起点は除外)。EUR・GBP・AUD のペアの {lc['e_vs_live_n']:,} 起点では、"
         f"見込みのずれは本番の season.roll_shift と最大 {lc['e_vs_live_max']:.1e} bp の差でした。",
    ]
    if sig:
        L.append("- **売買のシグナル:** research_trade.signal (+ 見送りの判定) と本番の trade.rule_signal は、1時間足 "
                 f"{sum(x['bars'] for x in sig['1h'].values()):,} 本で {sum(x['mismatch'] for x in sig['1h'].values())} 本、ロンドン日足 "
                 f"{sum(x['bars'] for x in sig['1d'].values()):,} 本で {sum(x['mismatch'] for x in sig['1d'].values())} 本が不一致。")
    for tf in ("1h", "1d"):
        x = bi[tf]
        L.append(f"- **レンジ ({'1時間足' if tf == '1h' else '日足'}):** 抜き取り {sum(v['n'] for v in x.values())} 起点で engine.sigma_steps との"
                 f"相対差は最大 {max(max(v['max_rel_diff'].values()) if isinstance(v['max_rel_diff'], dict) else (v['max_rel_diff'] or 0) for v in x.values()):.1e}、"
                 f"目標時刻の不一致 {sum(v['target_mismatch'] for v in x.values())} 件。k の再現と learning.learn_arrays の差は最大 "
                 f"{max(res['bands'][tf][h]['base']['replay_check'] for h in res['bands'][tf]):.4f}。")
    rep = res.get("reproduce")
    if rep:
        m = rep["match"]
        L.append("- **7ペアでの再現:** このモジュールのコードを元の7ペアに以前の規則で当てはめると、既存の報告の数字と "
                 + "、".join(f"{TEST_NO[k]}. {'一致' if v else '不一致'}" for k, v in m.items())
                 + " (1・2: rollover.json の Dukascopy、3: signals.json、4・5: bands.json)。")
    L.append("")
    return L


def report(res: dict) -> str:
    L = ["# ホールドアウトの6ペアでの確認: 採用したルールは設計に使っていないペアでも成り立つか", "",
         "本番の7ペアで調整期間・検証期間に分けて選んだルールを、どの選択にも使っていない6ペア (EURGBP・EURAUD・GBPAUD・USDCHF・USDCAD・"
         "NZDUSD) の約20年分の1時間足に、本番の設定のまま当てはめました。", "", res["prereg"].rstrip(), ""]
    L += _summary(res)
    L += _bullets(res) + [""] + _notes(res) + _decisions(res)
    L += _calls_section(res) + _promote_section(res) + _trade_section(res) + _shape_section(res) + _planb_section(res)
    if res["planb"]["pass"]:
        L += _planb_code()
    L += _method_section(res) + _check_section(res)
    L += ["## 再実行", "", "```bash",
          "python -m aifx.research_holdout   # research/holdout.md と research/holdout.json を作成 (2並列、"
          f"約{max(1, round(res['seconds'] / 60))}分)", "```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    run()
