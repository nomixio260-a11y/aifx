"""Command line: ``aifx cycle | serve | verify | audit | export | status | research | backtest``."""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from pathlib import Path

from .data import PAIRS, SyntheticMarket, get_pair


def _pairs_arg(value: str) -> list[str]:
    if value.lower() == "all":
        return list(PAIRS)
    return [get_pair(v).code for v in value.split(",") if v.strip()]


def _cycle_kwargs(args) -> dict:
    kw = {"pairs": args.pairs, "collect_news": not args.no_news}
    if args.synthetic:
        kw["market"] = SyntheticMarket(seed=1)
        kw["collect_news"] = False
    return kw


def cmd_cycle(args) -> int:
    from .api import build_api, write_api
    from .audit import audit
    from .pipeline import run_cycle
    from .site import write_site

    report = run_cycle(args.state, **_cycle_kwargs(args))
    if args.audit:
        rep = audit(args.state, sample=args.audit)
        Path(args.state, "cache").mkdir(parents=True, exist_ok=True)
        Path(args.state, "cache", "audit.json").write_text(json.dumps(rep), encoding="utf-8")
        print(f"audit: {'OK' if rep['ok'] else 'MISMATCH'} ({rep['checked']} predictions recomputed)")
        if not rep["ok"]:
            return 3
    if args.site:
        write_site(args.site)
        write_api(build_api(args.state, mode="static", interval_min=args.interval), args.site)
        print(f"wrote {args.site}/index.html and {args.site}/api/")
    if not report.verify["ok"]:
        for p in report.verify["problems"][:20]:
            print(f"VERIFY {p['kind']}: {p['msg']}", file=sys.stderr)
        return 2
    return 0


def cmd_serve(args) -> int:
    from .server import serve

    serve(args.state, args.site or "site", host=args.host, port=args.port, interval_min=args.interval,
          cycle_kwargs=_cycle_kwargs(args))
    return 0


def cmd_verify(args) -> int:
    from .audit import verify
    from .ledger import Ledger

    rep = verify(args.state)
    if args.expect_head:
        seq, _, h = args.expect_head.partition(":")
        recs = Ledger(args.state).load(check=False).records
        seq = int(seq)
        if seq and (len(recs) < seq or recs[seq - 1]["hash"] != h):
            rep["ok"] = False
            rep["problems"].insert(0, {"kind": "history", "msg": f"record {seq} changed or disappeared since {h[:12]}"})
    print(json.dumps({k: rep[k] for k in ("ok", "records", "head", "counts", "n_problems")}, ensure_ascii=False))
    for p in rep["problems"][:30]:
        print(f"  {p['kind']}: {p['msg']}")
    return 0 if rep["ok"] else 2


def cmd_head(args) -> int:
    from .ledger import Ledger

    seq, h = Ledger(args.state).load(check=False).head
    print(f"{seq}:{h}")
    return 0


def cmd_audit(args) -> int:
    from .audit import audit, external_check

    rep = audit(args.state, sample=args.sample)
    Path(args.state, "cache").mkdir(parents=True, exist_ok=True)
    for r in rep["results"]:
        print(f"  seq {r['seq']:>6} {r['pair']} {r['tf']} {r['origin']}  {'OK' if r['ok'] else 'MISMATCH'}  {r['diffs']}")
    print(f"audit: {'OK' if rep['ok'] else 'MISMATCH'}; recomputed {rep['checked']}, "
          f"skipped {rep['skipped_other_version']} made by another model version")
    ok = rep["ok"]
    if args.external:
        from .data import YahooMarket
        ext = external_check(args.state, YahooMarket())
        Path(args.state, "cache", "external.json").write_text(json.dumps(ext), encoding="utf-8")
        print(f"external price check: {ext['ok']} ({ext['checked']} bars compared)")
        ok = ok and ext["ok"] is not False
    Path(args.state, "cache", "audit.json").write_text(json.dumps(rep), encoding="utf-8")
    return 0 if ok else 3


def cmd_export(args) -> int:
    from .api import build_api, write_api
    from .site import write_site

    write_site(args.site)
    write_api(build_api(args.state, mode=args.mode, interval_min=args.interval), args.site)
    print(f"wrote {args.site}/index.html and {args.site}/api/")
    return 0


def _w(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _w(text))


def cmd_status(args) -> int:
    from .api import build_api

    api = build_api(args.state)
    meta = api["meta.json"]
    print(f"cycle {meta['cycle_at']}  ledger {meta['ledger']['records']} records, "
          f"verify {'OK' if meta['ledger']['ok'] else 'FAILED'}")
    for s in meta["pairs"]:
        d = s["decimals"]
        line = f"{_pad(s['label'], 9)}{s['price']:>12.{d}f}  "
        for tf, rows in s["outlook"].items():
            line += "  ".join(f"{r['label']} {r['dir']} {r['p_up'] * 100:.0f}%" for r in rows) + "   "
        print(line)
    print("\nlive track record:")
    for tf, hs in meta["track"].items():
        for h, st in hs.items():
            if st["n"]:
                d = st["direction"] or {}
                rate = d.get("rate")
                print(f"  {tf} h={h:>2}: n={st['n']:>5}  direction {rate * 100 if rate is not None else math.nan:5.1f}%"
                      f"  RW skill {st['skill'] * 100 if st['skill'] is not None else math.nan:+.1f}%")
    return 0


def cmd_research(args) -> int:
    from . import history, research, research_intraday, research_trade

    if args.download:
        history.download()
        return 0
    if args.trade:
        research_trade.run()
        return 0
    if args.intraday:
        research_intraday.run(workers=args.workers)
        return 0
    research.run(workers=args.workers)
    return 0


def cmd_backtest(args) -> int:
    from . import backtest
    from .data import PAIRS
    from .engine import TIMEFRAMES
    from .pipeline import State
    from .timeutil import utcnow

    state = State.open(args.state)
    tfs = [TIMEFRAMES[k] for k in (args.timeframes or list(TIMEFRAMES))]
    rep = backtest.update(state, tfs, list(PAIRS), utcnow(), budget=args.budget, log=print)
    print(json.dumps(rep["rows"], ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aifx", description="FXチャート予測サーバー")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state", default="state", help="台帳とデータの保存先 (既定 state/)")
    run = argparse.ArgumentParser(add_help=False)
    run.add_argument("--pairs", type=_pairs_arg, default=list(PAIRS), help="通貨ペア (例: USDJPY,EURUSD / all)")
    run.add_argument("--no-news", action="store_true", help="ニュースと経済指標カレンダーを取得しない")
    run.add_argument("--synthetic", action="store_true", help="乱数で作った相場で動作確認 (ネット不要)")
    run.add_argument("--interval", type=float, default=15, help="更新間隔 (分)")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("cycle", parents=[common, run], help="1回分の処理 (取得・判定・学習・予測・検証)")
    c.add_argument("--site", help="WebページとAPIを書き出すディレクトリ")
    c.add_argument("--audit", type=int, default=0, help="予測を再計算して監査する件数")
    c.set_defaults(func=cmd_cycle)

    s = sub.add_parser("serve", parents=[common, run], help="サーバーとして常駐し、Webページを配信")
    s.add_argument("--site", default="site")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve, interval=5)

    v = sub.add_parser("verify", parents=[common], help="台帳の改ざん・時刻違反・未判定がないか検証")
    v.add_argument("--expect-head", help="以前の台帳の先頭 SEQ:HASH がそのまま残っているか確認")
    v.set_defaults(func=cmd_verify)

    hd = sub.add_parser("head", parents=[common], help="台帳の先頭 SEQ:HASH を表示")
    hd.set_defaults(func=cmd_head)

    a = sub.add_parser("audit", parents=[common], help="過去の予測を同じデータで再計算して一致を確認")
    a.add_argument("--sample", type=int, default=5)
    a.add_argument("--external", action="store_true", help="保存した価格を取得し直した価格と照合")
    a.set_defaults(func=cmd_audit)

    e = sub.add_parser("export", parents=[common], help="保存済みの状態からWebページとAPIを書き出す")
    e.add_argument("--site", default="site")
    e.add_argument("--mode", default="static", choices=["static", "server", "snapshot"])
    e.add_argument("--interval", type=float, default=15)
    e.set_defaults(func=cmd_export)

    st = sub.add_parser("status", parents=[common], help="最新の予測と実績をターミナルに表示")
    st.set_defaults(func=cmd_status)

    r = sub.add_parser("research", help="過去データで検証し research/report.md を作成 (台帳とは別)")
    r.add_argument("--download", action="store_true", help="過去の価格 (Yahoo Finance) と短期金利 (FRED) を data/history/ に取得")
    r.add_argument("--intraday", action="store_true", help="15分足・5分足 (直近約60日) の検証 (research/intraday.md)")
    r.add_argument("--trade", action="store_true", help="売買ルールの検証 (コスト込み、research/trade.md)")
    r.add_argument("--workers", type=int, default=4)
    r.set_defaults(func=cmd_research)

    b = sub.add_parser("backtest", parents=[common], help="保存済みの価格で直近のバックテストを更新 (cache/ に保存)")
    b.add_argument("--budget", type=int, default=100000, help="今回計算する起点の上限")
    b.add_argument("--timeframes", type=lambda v: v.split(","), help="例: 15m,1h")
    b.set_defaults(func=cmd_backtest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
