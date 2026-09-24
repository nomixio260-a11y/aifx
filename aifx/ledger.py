"""Append-only, hash-chained record of everything the forecaster did.

Every record carries ``seq`` (1, 2, 3 ...), ``prev`` (the previous record's
hash) and ``hash`` (SHA-256 of the record's canonical JSON without ``hash``).
Editing, deleting or reordering any record breaks the chain.

Raw inputs (price bars, news items, calendar events) live in append-only data
files. Each cycle writes a ``batch`` record holding, for every file it
touched, the new line count and a rolling hash over the appended bytes. That
ties each line of input data to a position in the chain, so a verifier can
tell exactly what data existed before any given prediction.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .timeutil import iso

GENESIS = "0" * 64


class LedgerError(RuntimeError):
    pass


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_hash(rec: dict) -> str:
    return sha256_hex(canonical({k: v for k, v in rec.items() if k != "hash"}))


def roll(prev_hex: str, data: bytes) -> str:
    return sha256_hex(bytes.fromhex(prev_hex) + data)


class Ledger:
    def __init__(self, state_dir: Path | str):
        self.dir = Path(state_dir) / "ledger"
        self.records: list[dict] = []

    # ---------------------------------------------------------------- reading
    def files(self) -> list[Path]:
        return sorted(self.dir.glob("*.jsonl")) if self.dir.exists() else []

    def load(self, check: bool = True) -> "Ledger":
        self.records = []
        for path in self.files():
            with open(path, "rb") as fh:
                for raw in fh:
                    if raw.strip():
                        self.records.append(json.loads(raw))
        if check:
            problems = chain_problems(self.records)
            if problems:
                raise LedgerError("ledger chain broken: " + "; ".join(problems[:5]))
        return self

    @property
    def head(self) -> tuple[int, str]:
        if not self.records:
            return 0, GENESIS
        return self.records[-1]["seq"], self.records[-1]["hash"]

    def of_type(self, kind: str) -> list[dict]:
        return [r for r in self.records if r["type"] == kind]

    def by_seq(self, seq: int) -> dict:
        rec = self.records[seq - 1]
        if rec["seq"] != seq:
            raise LedgerError(f"seq {seq} not at its position")
        return rec

    # ---------------------------------------------------------------- writing
    def append(self, rec: dict, at: datetime) -> dict:
        seq, prev = self.head
        if self.records and self.records[-1]["at"] > iso(at):
            raise LedgerError("ledger time must not go backwards")
        full = dict(rec)
        full.update({"seq": seq + 1, "prev": prev, "at": iso(at)})
        full["hash"] = record_hash(full)
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{full['at'][:7]}.jsonl"
        with open(path, "ab") as fh:
            fh.write(canonical(full) + b"\n")
        self.records.append(full)
        return full


def chain_problems(records: list[dict]) -> list[str]:
    problems = []
    prev = GENESIS
    last_at = ""
    for i, rec in enumerate(records, start=1):
        if rec.get("seq") != i:
            problems.append(f"record {i}: seq is {rec.get('seq')}")
        if rec.get("prev") != prev:
            problems.append(f"seq {i}: prev hash does not match the record before it")
        if record_hash(rec) != rec.get("hash"):
            problems.append(f"seq {i}: content does not match its hash")
        if rec.get("at", "") < last_at:
            problems.append(f"seq {i}: time goes backwards")
        prev = rec.get("hash")
        last_at = rec.get("at", "")
    return problems


class DataFiles:
    """Append-only data files whose state is committed through ``batch`` records."""

    def __init__(self, state_dir: Path | str, ledger: Ledger):
        self.root = Path(state_dir)
        self.ledger = ledger
        self.state: dict[str, tuple[int, str]] = {}   # path -> (lines, rolling hash)
        self.dirty: set[str] = set()
        for rec in ledger.of_type("batch"):
            for path, (lines, rh) in rec["files"].items():
                self.state[path] = (lines, rh)
        self._drop_uncommitted_tails()

    def _drop_uncommitted_tails(self) -> None:
        """A crashed cycle can leave lines that no batch record covers; they were
        never used by any committed record, so they are discarded."""
        for path in list(self.state) + [str(p.relative_to(self.root)) for p in self._all_data_paths()]:
            full = self.root / path
            if not full.exists():
                continue
            lines = self.state.get(path, (0, GENESIS))[0]
            with open(full, "rb") as fh:
                content = fh.read()
            parts = content.split(b"\n")
            have = len(parts) - 1 if content.endswith(b"\n") else len(parts)
            if have > lines:
                keep = b"".join(p + b"\n" for p in parts[:lines])
                with open(full, "wb") as fh:
                    fh.write(keep)

    def _all_data_paths(self) -> list[Path]:
        out = []
        for sub in ("prices", "news", "calendar"):
            d = self.root / sub
            if d.exists():
                out.extend(sorted(d.glob("*")))
        return out

    def lines(self, path: str) -> int:
        return self.state.get(path, (0, GENESIS))[0]

    def append_lines(self, path: str, lines: list[str]) -> None:
        if not lines:
            return
        data = "".join(line.rstrip("\n") + "\n" for line in lines).encode("utf-8")
        full = self.root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "ab") as fh:
            fh.write(data)
        count, rh = self.state.get(path, (0, GENESIS))
        self.state[path] = (count + len(lines), roll(rh, data))
        self.dirty.add(path)

    def read_lines(self, path: str) -> list[str]:
        """Committed-or-pending lines of a file (exact text, without newlines)."""
        full = self.root / path
        if not full.exists():
            return []
        with open(full, "rb") as fh:
            return fh.read().decode("utf-8").splitlines()

    def commit(self, at: datetime, extra: dict | None = None) -> dict | None:
        """Write a batch record covering every file appended since the last commit."""
        if not self.dirty:
            return None
        files = {p: list(self.state[p]) for p in sorted(self.dirty)}
        rec = {"type": "batch", "files": files}
        if extra:
            rec.update(extra)
        out = self.ledger.append(rec, at)
        self.dirty.clear()
        return out


def data_file_problems(state_dir: Path | str, ledger: Ledger) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    """Recompute every data file's rolling hash at each batch record.

    Returns (problems, commits) where commits maps path -> [(batch seq, line count)]
    so callers can tell which batch first contained a given line.
    """
    root = Path(state_dir)
    problems: list[str] = []
    commits: dict[str, list[tuple[int, int]]] = {}
    cache: dict[str, list[bytes]] = {}
    progress: dict[str, tuple[int, str]] = {}
    for rec in ledger.of_type("batch"):
        for path, (lines, rh) in rec["files"].items():
            if path not in cache:
                full = root / path
                if not full.exists():
                    problems.append(f"{path}: file missing")
                    cache[path] = []
                else:
                    with open(full, "rb") as fh:
                        cache[path] = [p + b"\n" for p in fh.read().split(b"\n")[:-1]]
            done, cur = progress.get(path, (0, GENESIS))
            if lines < done:
                problems.append(f"{path}: batch seq {rec['seq']} shrinks the file")
                continue
            chunk = b"".join(cache[path][done:lines])
            if len(cache[path]) < lines:
                problems.append(f"{path}: has {len(cache[path])} lines, batch seq {rec['seq']} committed {lines}")
                continue
            cur = roll(cur, chunk)
            if cur != rh:
                problems.append(f"{path}: lines {done + 1}-{lines} differ from what batch seq {rec['seq']} committed")
            progress[path] = (lines, rh)
            commits.setdefault(path, []).append((rec["seq"], lines))
    for path, rows in cache.items():
        committed = progress.get(path, (0, GENESIS))[0]
        if len(rows) > committed:
            problems.append(f"{path}: {len(rows) - committed} line(s) not covered by any batch")
    return problems, commits
