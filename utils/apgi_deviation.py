"""Deviation detection and exploratory labelling (APGI-REP-004).

Compares the parameters/analyses of a run with the registered protocol and
labels the output ``exploratory`` (with linked deviation records) when anything
differs from or is absent in the registration. Output is never silently
promoted to confirmatory.
"""

from typing import Any, Dict, List, Optional

from utils.apgi_protocol_spec import registered_ranges, verify_protocol_lock


def detect_deviations(
    parameters: Dict[str, float],
    analyses: Optional[List[str]] = None,
    registered_analyses: Optional[List[str]] = None,
    protocols_dir: Any = None,
) -> List[Dict[str, Any]]:
    """Return deviation records (empty list = fully within registration)."""
    out: List[Dict[str, Any]] = []
    ranges = registered_ranges(protocols_dir)
    for name, (lo, hi) in ranges.items():
        if name not in parameters:
            out.append({"kind": "missing_registered_parameter", "name": name})
        elif not (lo <= parameters[name] <= hi):
            out.append({"kind": "parameter_outside_registered_range", "name": name,
                        "value": parameters[name], "registered": [lo, hi]})
    for name in parameters:
        if name not in ranges:
            out.append({"kind": "unregistered_parameter", "name": name})
    if registered_analyses is not None:
        for a in analyses or []:
            if a not in registered_analyses:
                out.append({"kind": "unregistered_analysis", "name": a})
    return out


def label_output(
    parameters: Dict[str, float],
    analyses: Optional[List[str]] = None,
    registered_analyses: Optional[List[str]] = None,
    protocols_dir: Any = None,
) -> Dict[str, Any]:
    """Attach the confirmatory/exploratory label and the verified protocol lock."""
    lock = verify_protocol_lock(protocols_dir)  # raises if the registration is altered
    devs = detect_deviations(parameters, analyses, registered_analyses, protocols_dir)
    return {
        "status": "exploratory" if devs else "confirmatory_eligible",
        "deviations": devs,
        "protocol_lock": lock,
    }
