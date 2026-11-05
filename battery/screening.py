"""Effort, fraud and validity screening (S3-SCR-01.1-.8).

All functions are pure and return flags, never silent drops: a flagged session
stays in the dataset with its reasons so exclusion rates can be reported in the
dataset card (S3-SCR-01.8). Thresholds live in SCREENING_CONFIG and are stamped
on every result so a threshold change is a visible, versioned event.
"""

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional, Sequence

SCREENING_CONFIG: dict[str, Any] = {
    "version": "screening_config-1.0.0",
    "min_plausible_rt_ms": 150.0,  # S3-SCR-01.1
    "max_fast_fraction": 0.10,  # share of trials below the floor that raises the flag
    "invariant_min_trials": 20,  # S3-SCR-01.2
    "straightline_min_items": 8,  # S3-SCR-01.3
    "straightline_max_run_fraction": 0.90,
    "zigzag_min_alternations": 0.90,
    "max_failed_attention_checks": 2,  # S3-SCR-01.4 (>= 2 excluded from norms)
    "bot_min_form_seconds": 5.0,  # S3-SCR-01.7
    "bot_min_item_seconds": 0.5,
}


def _cfg(config: Optional[dict[str, Any]]) -> dict[str, Any]:
    return config or SCREENING_CONFIG


def flag_fast_responses(
    rts_ms: Sequence[float], config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """S3-SCR-01.1: count responses faster than the plausible floor."""
    cfg = _cfg(config)
    vals = [float(r) for r in rts_ms if r is not None and math.isfinite(float(r))]
    if not vals:
        return {"flag": False, "n": 0, "n_fast": 0, "fraction_fast": 0.0, "reason": "no_rts"}
    n_fast = sum(v < cfg["min_plausible_rt_ms"] for v in vals)
    frac = n_fast / len(vals)
    return {
        "flag": frac > cfg["max_fast_fraction"],
        "n": len(vals),
        "n_fast": n_fast,
        "fraction_fast": frac,
        "config_version": cfg["version"],
    }


def flag_invariant_responding(
    keys: Sequence[Any],
    confidence: Optional[Sequence[float]] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """S3-SCR-01.2: same key on every trial AND zero confidence variance."""
    cfg = _cfg(config)
    n = len(keys)
    same_key = n >= cfg["invariant_min_trials"] and len(set(keys)) == 1
    conf_const = True
    if confidence is not None:
        conf_const = len(confidence) > 0 and len(set(confidence)) == 1
    return {
        "flag": bool(same_key and conf_const),
        "same_key": same_key,
        "zero_confidence_variance": conf_const if confidence is not None else None,
        "n": n,
        "config_version": cfg["version"],
    }


def flag_straightlining(
    responses: Sequence[int], config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """S3-SCR-01.3: straight-lining (one answer dominates) and zig-zagging
    (strictly alternating between two values) in a questionnaire block."""
    cfg = _cfg(config)
    n = len(responses)
    if n < cfg["straightline_min_items"]:
        return {"straightline": False, "zigzag": False, "flag": False, "n": n,
                "reason": "too_few_items"}
    top = Counter(responses).most_common(1)[0][1] / n
    straight = top >= cfg["straightline_max_run_fraction"]
    alternations = sum(
        responses[i] != responses[i - 1] and (i < 2 or responses[i] == responses[i - 2])
        for i in range(1, n)
    )
    zigzag = (
        len(set(responses)) == 2
        and alternations / (n - 1) >= cfg["zigzag_min_alternations"]
    )
    return {"straightline": straight, "zigzag": zigzag, "flag": straight or zigzag,
            "n": n, "config_version": cfg["version"]}


def exclude_from_norms(
    failed_attention_checks: int, config: Optional[dict[str, Any]] = None
) -> bool:
    """S3-SCR-01.4: >= 2 failed attention checks removes a session from norms."""
    if failed_attention_checks < 0:
        raise ValueError("failed_attention_checks must be non-negative")
    return failed_attention_checks >= _cfg(config)["max_failed_attention_checks"]


def duplicate_key(device_fingerprint: str, email: str) -> str:
    """Salted-free SHA-256 over fingerprint + normalised email (flag key only)."""
    norm = email.strip().lower()
    return hashlib.sha256(f"{device_fingerprint}|{norm}".encode()).hexdigest()


def flag_duplicates(sessions: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """S3-SCR-01.5: flag (never block) sessions sharing a fingerprint+email hash.

    Input items need ``session_id``, ``device_fingerprint``, ``email``. Returns
    ``{session_id: {"duplicate": bool, "group": key, "group_size": n}}``.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for s in sessions:
        groups[duplicate_key(s["device_fingerprint"], s["email"])].append(s["session_id"])
    out: dict[str, dict[str, Any]] = {}
    for key, ids in groups.items():
        for sid in ids:
            out[sid] = {"duplicate": len(ids) > 1, "group": key, "group_size": len(ids)}
    return out


def select_norm_sessions(
    sessions: Iterable[dict[str, Any]],
) -> dict[str, list[str]]:
    """S3-SCR-01.6: one norm-contributing session per participant per battery
    version - the earliest eligible one. Input needs ``session_id``,
    ``participant_id``, ``battery_version``, ``started_at`` (sortable) and
    ``norm_eligible`` (bool). Returns ``{"norm": [...], "not_norm": [...]}``."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    rest: list[str] = []
    ordered = sorted(sessions, key=lambda s: (s["started_at"], s["session_id"]))
    for s in ordered:
        if not s.get("norm_eligible", False):
            rest.append(s["session_id"])
            continue
        key = (s["participant_id"], s["battery_version"])
        if key in best:
            rest.append(s["session_id"])
        else:
            best[key] = s
    return {"norm": [s["session_id"] for s in best.values()], "not_norm": rest}


def screen_form_submission(
    form: dict[str, Any], config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """S3-SCR-01.7: form-layer bot screen. Looks at a hidden honeypot field,
    total completion time and per-item latency; flags, does not block."""
    cfg = _cfg(config)
    reasons: list[str] = []
    if str(form.get("honeypot", "")).strip():
        reasons.append("honeypot_filled")
    total = form.get("total_seconds")
    if total is None or not math.isfinite(float(total)) or float(total) < cfg["bot_min_form_seconds"]:
        reasons.append("form_too_fast")
    item_times = form.get("item_seconds") or []
    if item_times and sum(t < cfg["bot_min_item_seconds"] for t in item_times) / len(item_times) > 0.5:
        reasons.append("items_too_fast")
    return {"flag": bool(reasons), "reasons": reasons, "config_version": cfg["version"]}


def screen_session(session: dict[str, Any], config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Run every per-session check and return one record with all reasons."""
    cfg = _cfg(config)
    reasons: list[str] = []
    fast = flag_fast_responses(session.get("rts_ms", []), cfg)
    if fast["flag"]:
        reasons.append("fast_responses")
    inv = flag_invariant_responding(session.get("keys", []), session.get("confidence"), cfg)
    if inv["flag"]:
        reasons.append("invariant_responding")
    sl = flag_straightlining(session.get("questionnaire", []), cfg)
    if sl["flag"]:
        reasons.append("straightlining" if sl["straightline"] else "zigzag")
    failed = int(session.get("failed_attention_checks", 0))
    excluded = exclude_from_norms(failed, cfg)
    if excluded:
        reasons.append("failed_attention_checks")
    if "form" in session:
        bot = screen_form_submission(session["form"], cfg)
        reasons.extend(bot["reasons"])
    return {
        "session_id": session.get("session_id"),
        "reasons": reasons,
        "norm_eligible": not excluded and not any(
            r in reasons for r in ("fast_responses", "invariant_responding", "honeypot_filled")
        ),
        "config_version": cfg["version"],
    }


def exclusion_report(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """S3-SCR-01.8: exclusion rates, with denominators, for the dataset card."""
    n = len(records)
    counts: Counter = Counter()
    for r in records:
        counts.update(set(r.get("reasons", [])))
    return {
        "n_sessions": n,
        "n_norm_eligible": sum(bool(r.get("norm_eligible")) for r in records),
        "rates": {k: {"count": v, "rate": v / n if n else None} for k, v in sorted(counts.items())},
        "denominator": "all screened sessions",
        "note": "Observed rates only; no pilot or study data are implied.",
    }
