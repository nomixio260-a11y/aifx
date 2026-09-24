"""The forecasting server: runs cycles on a schedule and serves the web client.

    GET  /                 web client
    GET  /api/*.json       API documents (rewritten after every cycle)
    GET  /api/status       scheduler state (running, last/next run, last error)
    POST /api/refresh      run a cycle now (at most once a minute)

All computation happens here; the browser only displays the API.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import traceback
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

from .api import build_api, write_api
from .audit import audit
from .pipeline import run_cycle
from .site import write_site
from .timeutil import iso, utcnow

MIN_REFRESH_GAP = 60.0


class Scheduler:
    def __init__(self, state_dir: Path, site_dir: Path, interval_min: float, cycle_kwargs: dict,
                 audit_every: int = 6):
        self.state_dir = state_dir
        self.site_dir = site_dir
        self.interval = interval_min * 60
        self.cycle_kwargs = cycle_kwargs
        self.audit_every = audit_every
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.running = False
        self.cycles = 0
        self.last_started: datetime | None = None
        self.last_finished: datetime | None = None
        self.last_error: str | None = None
        self.next_run: datetime = utcnow()

    def status(self) -> dict:
        return {
            "running": self.running,
            "cycles": self.cycles,
            "last_started": iso(self.last_started) if self.last_started else None,
            "last_finished": iso(self.last_finished) if self.last_finished else None,
            "last_error": self.last_error,
            "next_run": iso(self.next_run),
            "interval_min": self.interval / 60,
            "server_time": iso(utcnow()),
        }

    def request_refresh(self) -> tuple[bool, str]:
        if self.running:
            return False, "更新中です"
        if self.last_started and (utcnow() - self.last_started).total_seconds() < MIN_REFRESH_GAP:
            return False, "直前に更新したばかりです。1分ほど待ってください"
        self.wake.set()
        return True, "更新を開始しました"

    def run_once(self) -> None:
        with self.lock:
            self.running = True
            self.last_started = utcnow()
            try:
                run_cycle(self.state_dir, **self.cycle_kwargs)
                self.cycles += 1
                if self.cycles % self.audit_every == 1:
                    rep = audit(self.state_dir)
                    (self.state_dir / "cache" / "audit.json").write_text(json.dumps(rep), encoding="utf-8")
                write_api(build_api(self.state_dir, mode="server", interval_min=self.interval / 60), self.site_dir)
                self.last_error = None
            except Exception as exc:  # keep serving the last good API
                self.last_error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                self.running = False
                self.last_finished = utcnow()
                self.next_run = self.last_finished + timedelta(seconds=self.interval)

    def loop(self) -> None:
        while True:
            self.run_once()
            wait = max(0.0, (self.next_run - utcnow()).total_seconds())
            self.wake.wait(timeout=wait)
            self.wake.clear()


class Handler(http.server.SimpleHTTPRequestHandler):
    scheduler: Scheduler

    def end_headers(self):
        if self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/status":
            return self._json(200, self.scheduler.status())
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] == "/api/refresh":
            ok, msg = self.scheduler.request_refresh()
            return self._json(202 if ok else 429, {"accepted": ok, "message": msg, **self.scheduler.status()})
        self.send_error(404)

    def log_message(self, fmt, *args):  # quieter console
        if "/api/status" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def serve(state_dir: Path | str, site_dir: Path | str, host: str = "127.0.0.1", port: int = 8000,
          interval_min: float = 5.0, cycle_kwargs: dict | None = None) -> None:
    state_dir, site_dir = Path(state_dir), Path(site_dir)
    write_site(site_dir)
    sched = Scheduler(state_dir, site_dir, interval_min, cycle_kwargs or {})
    threading.Thread(target=sched.loop, daemon=True).start()
    handler = partial(Handler, directory=str(site_dir))
    Handler.scheduler = sched
    with http.server.ThreadingHTTPServer((host, port), handler) as httpd:
        print(f"serving http://{host}:{port}/  (cycles every {interval_min:g} min; Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        time.sleep(0)
