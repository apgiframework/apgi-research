"""Dataset provenance records (APGI-DAT-004): source, version, checksum,
licence and transformation history attached to every dataset and verifiable."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REQUIRED = ("dataset_id", "source", "version", "license", "sha256", "transformations")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_record(path: Path | str, dataset_id: str, source: str, version: str, license: str,
                transformations: Optional[List[Dict[str, Any]]] = None,
                parents: Optional[List[str]] = None) -> Dict[str, Any]:
    if not all([dataset_id, source, version, license]):
        raise ValueError("dataset_id, source, version and license are all required")
    return {"dataset_id": dataset_id, "source": source, "version": version, "license": license,
            "sha256": sha256_file(path), "transformations": list(transformations or []),
            "parents": list(parents or [])}


def verify_record(record: Dict[str, Any], path: Path | str) -> List[str]:
    errors = [f"missing {k}" for k in REQUIRED if not record.get(k) and k != "transformations"]
    if "transformations" not in record:
        errors.append("missing transformations")
    if not errors and sha256_file(path) != record["sha256"]:
        errors.append("checksum mismatch")
    for i, t in enumerate(record.get("transformations", [])):
        for k in ("step", "software_version"):
            if not t.get(k):
                errors.append(f"transformation {i}: missing {k}")
    return errors


def write_sidecar(record: Dict[str, Any], path: Path | str) -> Path:
    side = Path(str(path) + ".provenance.json")
    side.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return side
