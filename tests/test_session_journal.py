import json

import pytest

from utils.session_journal import (JournalCorruptionError, OutOfOrderError,
                                   SessionJournal)


def apply(state, rec):
    state = dict(state)
    state["n"] = state.get("n", 0) + rec["payload"].get("inc", 0)
    return state


def test_append_resume_and_idempotency(tmp_path):
    j = SessionJournal(tmp_path / "s.jsonl", "S1")
    j.append("e1", "trial", {"inc": 1})
    again = j.append("e1", "trial", {"inc": 99})  # retry: no duplicate
    assert again["payload"]["inc"] == 1 and j.seq == 1
    j.append("e2", "trial", {"inc": 2})
    j.checkpoint("c1", {"n": 3})
    j.append("e3", "trial", {"inc": 4})
    reopened = SessionJournal(tmp_path / "s.jsonl", "S1")
    assert reopened.seq == 4 and reopened.resume(apply) == {"n": 7}


def test_checkpoint_resume_equals_full_replay(tmp_path):
    j = SessionJournal(tmp_path / "s.jsonl", "S1")
    for i in range(5):
        j.append(f"e{i}", "trial", {"inc": i})
    j.checkpoint("c", {"n": sum(range(5))})
    j.append("e9", "trial", {"inc": 10})
    full = SessionJournal(tmp_path / "s.jsonl", "S1")
    ckpt = full.resume(apply)
    manual = sum(range(5)) + 10
    assert ckpt["n"] == manual


def test_torn_final_write_is_dropped_not_fatal(tmp_path):
    p = tmp_path / "s.jsonl"
    j = SessionJournal(p, "S1")
    j.append("e1", "trial", {"inc": 1})
    with open(p, "ab") as fh:
        fh.write(b'{"seq":2,"sess')  # crash mid-write
    r = SessionJournal(p, "S1")
    assert r.seq == 1
    r.append("e2", "trial", {"inc": 2})
    assert SessionJournal(p, "S1").seq == 2


def test_tamper_gap_and_foreign_session_detected(tmp_path):
    p = tmp_path / "s.jsonl"
    j = SessionJournal(p, "S1")
    for i in range(3):
        j.append(f"e{i}", "trial", {"inc": i})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["payload"]["inc"] = 100
    tampered = tmp_path / "t.jsonl"
    tampered.write_text("\n".join([lines[0], json.dumps(rec), lines[2]]) + "\n")
    with pytest.raises(JournalCorruptionError, match="hash chain"):
        SessionJournal(tampered, "S1")
    gap = tmp_path / "g.jsonl"
    gap.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(JournalCorruptionError, match="gap"):
        SessionJournal(gap, "S1")
    with pytest.raises(JournalCorruptionError, match="foreign"):
        SessionJournal(p, "OTHER")


def test_out_of_order_rejected(tmp_path):
    j = SessionJournal(tmp_path / "s.jsonl", "S1")
    j.append("e1", "trial", {}, expected_seq=1)
    with pytest.raises(OutOfOrderError):
        j.append("e3", "trial", {}, expected_seq=3)
    assert j.seq == 1


def test_nan_payload_refused(tmp_path):
    j = SessionJournal(tmp_path / "s.jsonl", "S1")
    with pytest.raises(ValueError):
        j.append("e1", "trial", {"x": float("nan")})
    assert j.seq == 0
