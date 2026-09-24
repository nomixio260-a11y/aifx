"""Command line interface: ``aifx forecast | build | serve``."""

from __future__ import annotations

import argparse
import functools
import http.server
import sys
import threading
import time
import unicodedata
from pathlib import Path

from .data import PAIRS, get_pair, load_prices, synthetic_prices
from .forecast import build_bundle, build_report
from .site import render_fragment, write_site


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _ljust(text: str, width: int) -> str:
    return text + " " * max(0, width - _width(text))


def _rjust(text: str, width: int) -> str:
    return " " * max(0, width - _width(text)) + text


def _pairs_arg(value: str) -> list[str]:
    if value.lower() == "all":
        return list(PAIRS)
    return [get_pair(v).code for v in value.split(",") if v.strip()]


def _reports(args) -> list[dict]:
    reports = []
    for i, code in enumerate(args.pairs):
        pair = get_pair(code)
        if args.synthetic:
            df, source = synthetic_prices(n=1200, start_price=150.0 if pair.quote == "JPY" else 1.1, seed=i), "synthetic"
        else:
            df, source = load_prices(pair, cache_dir=args.cache, offline=args.offline, years=args.years)
        t0 = time.time()
        reports.append(build_report(pair, df, source, horizon=args.horizon))
        print(f"  {pair.label:8s} {len(df):5d} bars  source={source:18s} {time.time() - t0:5.2f}s", file=sys.stderr)
    return reports


def cmd_forecast(args) -> int:
    for r in _reports(args):
        d = r["decimals"]
        print(f"\n{r['label']} ({r['name']})  最新 {r['last_close']:.{d}f}  [{r['last_date']}]")
        print("  " + _ljust("期間", 8) + _rjust("予測", 12) + _rjust("変化(pips)", 12)
              + _rjust("上昇確率", 10) + "   80%レンジ")
        for o in r["outlook"]:
            print(
                "  " + _ljust(f"{o['h']}日後", 8) + f"{o['price']:>12.{d}f}{o['change_pips']:>+12.1f}"
                f"{o['p_up'] * 100:>9.0f}%   {o['lo80']:.{d}f} – {o['hi80']:.{d}f}  {o['label']}"
            )
        m = r["backtest"]["metrics"]["ensemble"]
        last = str(r["backtest"]["horizons"][-1])
        hit = m[last]["hit"]
        print(
            f"  検証: {last}日後の方向的中率 {hit * 100:.0f}% / RW比誤差改善 {m[last]['skill'] * 100:+.1f}%"
            f" ({r['backtest']['origins']}回)"
        )
    return 0


def cmd_build(args) -> int:
    bundle = build_bundle(_reports(args), horizon=args.horizon)
    index = write_site(bundle, args.out)
    print(f"wrote {index}")
    if args.fragment:
        Path(args.fragment).parent.mkdir(parents=True, exist_ok=True)
        Path(args.fragment).write_text(render_fragment(bundle), encoding="utf-8")
        print(f"wrote {args.fragment}")
    return 0


def cmd_serve(args) -> int:
    cmd_build(args)

    def refresher():
        while True:
            time.sleep(args.refresh * 60)
            try:
                cmd_build(args)
            except Exception as exc:  # keep serving the last good build
                print(f"refresh failed: {exc}", file=sys.stderr)

    if args.refresh > 0:
        threading.Thread(target=refresher, daemon=True).start()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.out))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        print(f"serving http://{args.host}:{args.port}/  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aifx", description="FXチャート予測 (日足・最大20営業日先)")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pairs", type=_pairs_arg, default=list(PAIRS),
                        help="通貨ペア (例: USDJPY,EURUSD / all)。既定は全7ペア")
    common.add_argument("--horizon", type=int, default=20, help="予測する営業日数 (既定 20)")
    common.add_argument("--years", type=int, default=5, help="取得する履歴の年数 (既定 5)")
    common.add_argument("--cache", default="data/cache", help="価格キャッシュのディレクトリ")
    common.add_argument("--offline", action="store_true", help="ネットワークを使わずキャッシュだけで計算")
    common.add_argument("--synthetic", action="store_true", help="乱数で作った価格で動作確認 (ネット不要)")
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("forecast", parents=[common], help="予測をターミナルに表示")
    f.add_argument("pair_codes", nargs="*", help="通貨ペア (省略時は --pairs)")
    f.set_defaults(func=cmd_forecast)

    b = sub.add_parser("build", parents=[common], help="ダッシュボード (HTML) を生成")
    b.add_argument("--out", default="site", help="出力ディレクトリ (既定 site/)")
    b.add_argument("--fragment", help="<html>ラッパー無しのHTML断片も書き出すパス")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("serve", parents=[common], help="ダッシュボードを生成してローカルで配信")
    s.add_argument("--out", default="site")
    s.add_argument("--fragment", default=None)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--refresh", type=float, default=0, help="データを再取得する間隔 (分)。0で無効")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "pair_codes", None):
        args.pairs = [get_pair(c).code for c in args.pair_codes]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
