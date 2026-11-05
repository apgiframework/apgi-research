"""Construct-to-task bridge validation (S3-BAT-05): no task enters the battery
until every construct it claims has a complete bridge row."""

import json
import pathlib
from typing import Any, Iterable, Optional

BRIDGE_PATH = pathlib.Path(__file__).with_name("construct_bridge.json")
REQUIRED = ("row_id", "construct", "symbol", "task", "observables", "estimator",
            "testable_v0_1", "reporting_scale", "limitations")


def load_bridge(path: Optional[pathlib.Path] = None) -> dict[str, Any]:
    return json.loads(pathlib.Path(path or BRIDGE_PATH).read_text(encoding="utf-8"))


def bridge_errors(bridge: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = bridge.get("rows", [])
    ids = [r.get("row_id") for r in rows]
    if len(set(ids)) != len(ids):
        errors.append("duplicate row_id")
    for r in rows:
        rid = r.get("row_id", "<missing row_id>")
        for key in REQUIRED:
            if key not in r:
                errors.append(f"{rid}: missing {key}")
        if r.get("testable_v0_1") is True:
            if not r.get("task") or not r.get("observables") or not r.get("estimator"):
                errors.append(f"{rid}: testable row needs task, observables and estimator")
        if r.get("testable_v0_1") is False and r.get("task") and not r.get("gate"):
            errors.append(f"{rid}: untestable row with a task must name the gate that unblocks it")
        if not str(r.get("limitations", "")).strip():
            errors.append(f"{rid}: limitations must be stated")
    return errors


def tasks_cleared_for_battery(bridge: dict[str, Any], tasks: Iterable[str]) -> dict[str, bool]:
    """A task is cleared only if it has >= 1 row and every one of its rows is
    testable and complete."""
    bad = {r["row_id"] for r in bridge["rows"] if bridge_errors({"rows": [r]})}
    out: dict[str, bool] = {}
    for t in tasks:
        rows = [r for r in bridge["rows"] if r.get("task") == t]
        out[t] = bool(rows) and all(r["testable_v0_1"] and r["row_id"] not in bad for r in rows)
    return out
