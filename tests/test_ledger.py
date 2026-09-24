import json
from datetime import datetime, timedelta, timezone

import pytest

from aifx.ledger import DataFiles, Ledger, LedgerError, chain_problems, data_file_problems

UTC = timezone.utc
T0 = datetime(2026, 9, 24, 6, 0, tzinfo=UTC)


def build(tmp_path, n=5):
    ledger = Ledger(tmp_path).load()
    files = DataFiles(tmp_path, ledger)
    for i in range(n):
        files.append_lines("prices/X_1h.csv", [f"row{i}a", f"row{i}b"])
        files.commit(T0 + timedelta(minutes=i))
        ledger.append({"type": "prediction", "i": i}, T0 + timedelta(minutes=i))
    return ledger


def ledger_lines(tmp_path):
    path = next((tmp_path / "ledger").glob("*.jsonl"))
    return path, path.read_text().splitlines()


def test_chain_round_trip(tmp_path):
    build(tmp_path)
    again = Ledger(tmp_path).load()
    assert len(again.records) == 10
    assert not chain_problems(again.records)
    assert data_file_problems(tmp_path, again)[0] == []


@pytest.mark.parametrize("attack", ["edit", "delete", "swap", "insert"])
def test_any_change_to_history_is_detected(tmp_path, attack):
    build(tmp_path)
    path, lines = ledger_lines(tmp_path)
    if attack == "edit":
        rec = json.loads(lines[3]); rec["i"] = 99; lines[3] = json.dumps(rec)
    elif attack == "delete":
        del lines[3]
    elif attack == "swap":
        lines[3], lines[5] = lines[5], lines[3]
    else:
        lines.insert(4, lines[3])
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        Ledger(tmp_path).load()


def test_recomputing_the_hash_does_not_hide_an_edit(tmp_path):
    from aifx.ledger import record_hash
    build(tmp_path)
    path, lines = ledger_lines(tmp_path)
    rec = json.loads(lines[3]); rec["i"] = 99; rec["hash"] = record_hash(rec); lines[3] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    problems = chain_problems(Ledger(tmp_path).load(check=False).records)
    assert any("prev hash" in p for p in problems)


def test_data_file_edits_and_truncation_are_detected(tmp_path):
    build(tmp_path)
    data = tmp_path / "prices" / "X_1h.csv"
    original = data.read_text()
    data.write_text(original.replace("row2a", "row2A"))
    assert data_file_problems(tmp_path, Ledger(tmp_path).load())[0]
    data.write_text("\n".join(original.splitlines()[:4]) + "\n")
    assert data_file_problems(tmp_path, Ledger(tmp_path).load())[0]


def test_uncommitted_tail_is_discarded_on_open(tmp_path):
    ledger = build(tmp_path, n=2)
    files = DataFiles(tmp_path, ledger)
    files.append_lines("prices/X_1h.csv", ["crashed-cycle"])  # never committed
    DataFiles(tmp_path, Ledger(tmp_path).load())
    assert "crashed-cycle" not in (tmp_path / "prices" / "X_1h.csv").read_text()


def test_time_cannot_go_backwards(tmp_path):
    ledger = build(tmp_path, n=1)
    with pytest.raises(LedgerError):
        ledger.append({"type": "prediction"}, T0 - timedelta(hours=1))
