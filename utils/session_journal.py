"""Reliable session journal with checkpoint resume (S2-ING-02, S2-ING-02.3,
APGI-PRO-003).

Append-only JSON-lines file. Every record carries a monotonically increasing
``seq``, a client-supplied ``event_id`` for idempotency, and a SHA-256 chain
over (prev_hash, payload) so tampering, reordering or a lost record is
detectable. Writes are flushed and fsynced before ``append`` returns. A torn
final line (crash mid-write) is tolerated on open and truncated; any other
corruption raises ``JournalCorruptionError`` rather than being skipped.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

GENESIS = "0" * 64


class JournalCorruptionError(RuntimeError):
    pass


class OutOfOrderError(ValueError):
    pass


def _digest(prev: str, body: dict[str, Any]) -> str:
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(f"{prev}|{canon}".encode()).hexdigest()


class SessionJournal:
    def __init__(self, path: Path | str, session_id: str) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.seq = 0
        self.head = GENESIS
        self.event_ids: set[str] = set()
        self.records: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._recover()

    # -- recovery ---------------------------------------------------------
    def _recover(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        lines = raw.split(b"\n")
        torn = lines[-1] != b""  # no trailing newline => last write incomplete
        complete = lines[:-1]
        if torn:
            self._truncate(len(raw) - len(lines[-1]))
        for i, line in enumerate(complete, 1):
            try:
                rec = json.loads(line)
                body = {k: rec[k] for k in ("seq", "session_id", "event_id", "kind", "payload")}
            except (ValueError, KeyError) as exc:
                raise JournalCorruptionError(f"line {i} unreadable: {exc}") from exc
            if rec["session_id"] != self.session_id:
                raise JournalCorruptionError(f"line {i}: foreign session_id")
            if rec["seq"] != self.seq + 1:
                raise JournalCorruptionError(f"line {i}: sequence gap ({self.seq} -> {rec['seq']})")
            if rec.get("hash") != _digest(self.head, body):
                raise JournalCorruptionError(f"line {i}: hash chain broken")
            self.seq, self.head = rec["seq"], rec["hash"]
            self.event_ids.add(rec["event_id"])
            self.records.append(rec)

    def _truncate(self, size: int) -> None:
        with open(self.path, "r+b") as fh:
            fh.truncate(size)
            fh.flush()
            os.fsync(fh.fileno())

    # -- writing ----------------------------------------------------------
    def append(self, event_id: str, kind: str, payload: dict[str, Any],
               expected_seq: Optional[int] = None) -> dict[str, Any]:
        """Durably append one event. A repeated ``event_id`` is a no-op returning
        the original record (idempotent retry). ``expected_seq`` lets a caller
        detect out-of-order delivery: the next seq must equal it."""
        if event_id in self.event_ids:
            return next(r for r in self.records if r["event_id"] == event_id)
        if expected_seq is not None and expected_seq != self.seq + 1:
            raise OutOfOrderError(f"expected seq {expected_seq}, journal is at {self.seq}")
        body = {"seq": self.seq + 1, "session_id": self.session_id, "event_id": event_id,
                "kind": kind, "payload": payload}
        rec = {**body, "hash": _digest(self.head, body)}
        line = json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        with open(self.path, "ab") as fh:
            fh.write(line.encode())
            fh.flush()
            os.fsync(fh.fileno())
        self.seq, self.head = rec["seq"], rec["hash"]
        self.event_ids.add(event_id)
        self.records.append(rec)
        return rec

    def checkpoint(self, event_id: str, state: dict[str, Any]) -> dict[str, Any]:
        """Record a full state snapshot; resume starts from the latest one."""
        return self.append(event_id, "checkpoint", {"state": state, "through_seq": self.seq})

    # -- resume -----------------------------------------------------------
    def resume(self, apply: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
               initial: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Rebuild state: latest checkpoint, then replay later non-checkpoint
        events through ``apply(state, record) -> state``. Replaying the journal
        from scratch must give the same state (checked by tests)."""
        state = dict(initial or {})
        start = 0
        for i in range(len(self.records) - 1, -1, -1):
            if self.records[i]["kind"] == "checkpoint":
                state = json.loads(json.dumps(self.records[i]["payload"]["state"]))
                start = i + 1
                break
        for rec in self.records[start:]:
            if rec["kind"] != "checkpoint":
                state = apply(state, rec)
        return state
