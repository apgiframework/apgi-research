"""Participant consent and scope enforcement (APGI-CON-001).

Four consent purposes are recorded per participant. Every operation that
ingests, processes, scores or exports research data passes through
``ConsentLedger.require`` (or the ``requires_consent`` decorator): absent,
withdrawn, declined, version-mismatched or out-of-scope requests raise
``ConsentError`` and are logged, and nothing is persisted for a declined
purpose.
"""

import functools
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class Purpose(str, Enum):
    """The four registered consent purposes. Names are neutral placeholders
    pending the approved consent text; the enforcement logic does not depend on
    their wording."""

    CORE_RESEARCH = "core_research"
    PHYSIOLOGICAL_SIGNALS = "physiological_signals"
    DATA_SHARING = "data_sharing"
    RECONTACT = "recontact"


class ConsentError(PermissionError):
    pass


@dataclass
class ConsentRecord:
    participant_id: str
    consent_version: str
    granted: Dict[str, bool]
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    withdrawn_at: Optional[str] = None

    def __post_init__(self) -> None:
        missing = {p.value for p in Purpose} - set(self.granted)
        if missing:
            raise ValueError(f"every purpose needs an explicit decision; missing {sorted(missing)}")
        if not all(isinstance(v, bool) for v in self.granted.values()):
            raise ValueError("consent decisions must be booleans")


class ConsentLedger:
    def __init__(self, required_version: str) -> None:
        self.required_version = required_version
        self._records: Dict[str, ConsentRecord] = {}
        self.denials: List[Dict[str, Any]] = []

    def record(self, participant_id: str, consent_version: str, granted: Dict[str, bool]) -> ConsentRecord:
        rec = ConsentRecord(participant_id, consent_version, dict(granted))
        self._records[participant_id] = rec
        return rec

    def withdraw(self, participant_id: str) -> None:
        rec = self._records.get(participant_id)
        if rec is None:
            raise ConsentError(f"no consent record for {participant_id}")
        rec.withdrawn_at = datetime.now(timezone.utc).isoformat()

    def _deny(self, participant_id: str, purpose: str, operation: str, reason: str) -> None:
        self.denials.append({"participant_id": participant_id, "purpose": purpose,
                             "operation": operation, "reason": reason,
                             "at": datetime.now(timezone.utc).isoformat()})
        raise ConsentError(f"{operation}: {reason}")

    def require(self, participant_id: str, purpose: Purpose, operation: str) -> ConsentRecord:
        rec = self._records.get(participant_id)
        p = Purpose(purpose).value
        if rec is None:
            self._deny(participant_id, p, operation, "no consent recorded")
        assert rec is not None
        if rec.withdrawn_at:
            self._deny(participant_id, p, operation, "consent withdrawn")
        if rec.consent_version != self.required_version:
            self._deny(participant_id, p, operation,
                       f"consent version {rec.consent_version} != required {self.required_version}")
        if not rec.granted[p]:
            self._deny(participant_id, p, operation, f"purpose '{p}' declined")
        return rec

    def fingerprint(self, participant_id: str) -> str:
        """Stable hash of the consent state, stamped on derived records so lineage
        shows which consent authorised them."""
        rec = self._records[participant_id]
        return hashlib.sha256(json.dumps(rec.__dict__, sort_keys=True).encode()).hexdigest()


def requires_consent(ledger: ConsentLedger, purpose: Purpose, participant_arg: str = "participant_id") -> Callable:
    """Decorator: block the wrapped operation unless consent covers `purpose`."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect

            bound = inspect.signature(fn).bind(*args, **kwargs)
            bound.apply_defaults()
            ledger.require(bound.arguments[participant_arg], purpose, fn.__name__)
            return fn(*args, **kwargs)

        return wrapper

    return deco
