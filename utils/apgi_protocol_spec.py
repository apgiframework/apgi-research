"""
APGI protocol specification 1.1.0.

Reads the registered protocol files in ``protocols/`` (byte-identical to the
apgi-framework 1.1.0 release) and exposes the parts this research platform
consumes: the registered reduced model, cohort sizes, per-prediction decision
fields and the v1.1 three-way verdict.

Two v1.1 rules matter for code that reports outcomes:

- A failure to confirm is not falsification. A prediction is FALSIFIED only by
  its own registered falsifying rule; otherwise it is INDETERMINATE.
- kappa (ATP/bit) is uncalibrated (``null``) and never sets ignition
  steepness. No unregistered coefficient gets a legacy default
  (kappa = 100, alpha = 0.3, beta = 0.7 are gone).
"""

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROTOCOLS_DIR = Path(__file__).resolve().parents[1] / "protocols"

# Kept in step with apgi-framework (protocols/CHANGELOG.md, scripts/validate_protocols.py).
SPECIFICATION_VERSION = "1.1.0"


class ProtocolLockError(RuntimeError):
    """A protocol file does not match the registered lock (APGI-REP-001)."""


class VerdictConflictError(ValueError):
    """Contradictory or malformed evidence flags for a registered prediction (APGI-VAL-001)."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_lock(folder: Optional[Path] = None) -> Dict[str, str]:
    """Verify every protocol payload against ``MANIFEST.json`` before runtime use.

    Checks each manifest entry exists, matches its SHA-256, stays inside
    ``folder``, and that no unlisted ``protocol_*.json`` / ``VALIDATION.json`` /
    ``CHANGELOG.md`` file is present. Returns ``{relative_path: sha256}`` of the
    verified lock (retain it as run evidence). Raises ``ProtocolLockError``
    before any protocol content is parsed or used.
    """
    folder = Path(folder or PROTOCOLS_DIR).resolve()
    manifest_path = folder / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ProtocolLockError(f"protocol lock missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["files"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ProtocolLockError(f"protocol lock unreadable: {exc}") from exc
    if manifest.get("release") != SPECIFICATION_VERSION:
        raise ProtocolLockError(
            f"protocol lock release {manifest.get('release')!r} != {SPECIFICATION_VERSION!r}"
        )
    verified: Dict[str, str] = {}
    for entry in entries:
        name = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            raise ProtocolLockError("protocol lock has an invalid entry")
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ProtocolLockError(f"protocol lock path escapes folder: {name}")
        if not path.is_file():
            raise ProtocolLockError(f"locked protocol file missing: {name}")
        digest = _sha256_file(path)
        if digest != entry.get("sha256"):
            raise ProtocolLockError(f"locked protocol file altered: {name}")
        verified[name] = digest
    payload = {
        p.name
        for pattern in ("protocol_*.json", "VALIDATION.json", "CHANGELOG.md")
        for p in folder.glob(pattern)
    }
    unlocked = sorted(payload - set(verified))
    if unlocked:
        raise ProtocolLockError(f"protocol files not covered by the lock: {unlocked}")
    return verified


class Verdict(str, Enum):
    """v1.1 three-way decision for a registered prediction."""

    CONFIRMED = "confirmed"
    FALSIFIED = "falsified"
    INDETERMINATE = "indeterminate"


def classify_prediction(confirmed: bool, falsified: bool = False) -> Verdict:
    """Apply the v1.1 rule that a failure to confirm is not falsification.

    ``falsified`` must come from the prediction's own falsifying rule
    (``falsifying_evidence`` / ``falsifying_thresholds``), never from the
    negation of ``confirmed``.
    """
    if not isinstance(confirmed, bool) or not isinstance(falsified, bool):
        raise VerdictConflictError("confirmed and falsified must be booleans")
    if confirmed and falsified:
        raise VerdictConflictError(
            "prediction cannot be both confirmed and falsified; "
            "open a deviation record instead of choosing a verdict"
        )
    if confirmed:
        return Verdict.CONFIRMED
    if falsified:
        return Verdict.FALSIFIED
    return Verdict.INDETERMINATE


@dataclass
class SubPrediction:
    """One registered prediction (e.g. ``Pred 1.A``)."""

    id: str
    claim: str = ""
    status: str = ""
    confirming_evidence: str = ""
    falsifying_evidence: str = ""
    indeterminate_rule: str = ""
    confirming_thresholds: Dict[str, Any] = field(default_factory=dict)
    falsifying_thresholds: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubPrediction":
        return cls(
            id=data.get("id", ""),
            claim=data.get("claim", ""),
            status=data.get("status", ""),
            confirming_evidence=data.get("confirming_evidence", ""),
            falsifying_evidence=data.get("falsifying_evidence", ""),
            indeterminate_rule=data.get("indeterminate_rule", ""),
            confirming_thresholds=data.get("confirming_thresholds") or {},
            falsifying_thresholds=data.get("falsifying_thresholds") or {},
        )

    @property
    def is_confirmatory(self) -> bool:
        """True for predictions whose outcome carries a registered framework vote."""
        return self.status.startswith("confirmatory")


@dataclass
class ProtocolSpec:
    """A registered protocol file."""

    protocol_id: str
    title: str
    version: str
    study_status: str
    path: Path
    sub_predictions: List[SubPrediction]
    raw: Dict[str, Any] = field(repr=False)

    @classmethod
    def from_file(cls, path: Path, verify_lock: bool = True) -> "ProtocolSpec":
        """Parse one protocol file, verifying the registered lock first.

        ``verify_lock=False`` exists only for tooling that edits drafts (it
        marks nothing as registered); production callers must not pass it.
        """
        path = Path(path)
        if verify_lock:
            verified = verify_protocol_lock(path.parent)
            if path.name not in verified:
                raise ProtocolLockError(f"{path.name} is not part of the lock")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            protocol_id=data["protocol_id"],
            title=data.get("title", ""),
            version=data.get("version", ""),
            study_status=data.get("study_status", ""),
            path=path,
            sub_predictions=[
                SubPrediction.from_dict(p) for p in data.get("sub_predictions", [])
            ],
            raw=data,
        )

    @property
    def is_current_specification(self) -> bool:
        return self.version == SPECIFICATION_VERSION

    @property
    def participants(self) -> Dict[str, Any]:
        """Registered cohort block (n_target, n_analyzable_target, n_minimum, groups)."""
        return self.raw.get("participants", {})

    @property
    def apgi_parameters(self) -> Dict[str, Any]:
        """Registered reduced empirical model."""
        return self.raw.get("apgi_parameters", {})

    @property
    def submission_blocking_issues(self) -> List[str]:
        return self.raw.get("submission_blocking_issues", [])

    def get_prediction(self, pred_id: str) -> Optional[SubPrediction]:
        for pred in self.sub_predictions:
            if pred.id == pred_id:
                return pred
        return None


def _normalise_id(protocol: Any) -> str:
    """Accept ``3``, ``"P3"``, ``"P03"`` or ``"APGI-P03"``."""
    text = str(protocol).upper().replace("APGI-", "").lstrip("P")
    return f"APGI-P{int(text):02d}"


def load_all_protocols(protocols_dir: Optional[Path] = None) -> Dict[str, ProtocolSpec]:
    """Load every ``protocol_*.json`` keyed by protocol ID (``APGI-P00`` .. ``APGI-P07``)."""
    folder = protocols_dir or PROTOCOLS_DIR
    if not folder.is_dir():
        raise FileNotFoundError(f"Protocol specification folder not found: {folder}")
    verify_protocol_lock(folder)
    specs = (ProtocolSpec.from_file(p) for p in sorted(folder.glob("protocol_*.json")))
    return {spec.protocol_id: spec for spec in specs}


def load_protocol(protocol: Any, protocols_dir: Optional[Path] = None) -> ProtocolSpec:
    """Load one protocol by number or ID."""
    protocol_id = _normalise_id(protocol)
    specs = load_all_protocols(protocols_dir)
    if protocol_id not in specs:
        raise KeyError(f"Unknown protocol: {protocol}")
    return specs[protocol_id]


def registered_ranges(
    protocols_dir: Optional[Path] = None,
) -> Dict[str, Tuple[float, float]]:
    """Registered ranges shared by all protocols: theta_0, gamma_sig and beta_SM."""
    params = load_protocol(1, protocols_dir).apgi_parameters
    return {
        "theta_0": tuple(params["theta_0"]["registered_band"]),  # type: ignore[dict-item]
        "gamma_sig": tuple(params["gamma_sig"]["range"]),  # type: ignore[dict-item]
        "beta_SM": tuple(params["beta_SM"]["typical_range"]),  # type: ignore[dict-item]
    }


# APGIParameters field that plays the role of each registered quantity.
_REGISTRY_TO_MODEL = {"theta_0": "theta_0", "gamma_sig": "alpha", "beta_SM": "beta"}


@dataclass
class RangeCheck:
    """A model default compared with its registered range."""

    quantity: str
    model_field: str
    value: float
    registered: Tuple[float, float]

    @property
    def inside(self) -> bool:
        low, high = self.registered
        return low <= self.value <= high


def check_parameter_defaults(
    params: Any = None, protocols_dir: Optional[Path] = None
) -> List[RangeCheck]:
    """Compare ``APGIParameters`` defaults with the registered theta_0, gamma_sig and beta_SM.

    Reports only; it does not change the model. A default outside its registered
    range must not be used for a confirmatory fit.
    """
    if params is None:
        from APGI_System import APGIParameters

        params = APGIParameters()
    ranges = registered_ranges(protocols_dir)
    return [
        RangeCheck(name, field_name, float(getattr(params, field_name)), ranges[name])
        for name, field_name in _REGISTRY_TO_MODEL.items()
    ]
