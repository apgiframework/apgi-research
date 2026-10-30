"""Device quality tiers from a versioned tier_config (S3-BAT-01.6). Thresholds
mirror apgi-api `app/services/screening.py` TIER_CONFIG and must change
together; `tests/test_battery.py` pins the shared version string."""

from typing import Any, Optional

TIER_CONFIG: dict[str, Any] = {
    "version": "tier_config-1.0.0",
    "a_max_p95_timing_error_ms": 8.0,
    "b_max_p95_timing_error_ms": 20.0,
    "a_max_dropped_frame_fraction": 0.01,
    "b_max_dropped_frame_fraction": 0.05,
    "max_failed_attention_checks": 2,
}


def timing_error_distribution(errors_ms: list[float]) -> dict[str, float]:
    if not errors_ms:
        return {"n": 0}
    xs = sorted(abs(float(e)) for e in errors_ms)

    def q(p: float) -> float:
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]

    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": q(0.5),
        "p95": q(0.95),
        "max": xs[-1],
    }


def assign_tier(
    session: dict[str, Any], config: Optional[dict[str, Any]] = None
) -> tuple[str, list[str]]:
    """A: primary norms; B: sensitivity analyses; C: excluded from norms."""
    cfg = config or TIER_CONFIG
    p95 = timing_error_distribution(session.get("timing_errors_ms", [])).get("p95", 0.0)
    dropped = float(session.get("dropped_frame_fraction", 0.0))
    failed = int(session.get("failed_attention_checks", 0))
    c: list[str] = []
    if session.get("is_mobile"):
        c.append("mobile_device")
    if int(session.get("visibility_changes", 0)) > 0:
        c.append("tab_switch")
    if failed >= cfg["max_failed_attention_checks"]:
        c.append("failed_attention_checks")
    if p95 > cfg["b_max_p95_timing_error_ms"]:
        c.append("major_timing_jitter")
    if dropped > cfg["b_max_dropped_frame_fraction"]:
        c.append("dropped_frames")
    if c:
        return "C", c
    b: list[str] = []
    if p95 > cfg["a_max_p95_timing_error_ms"]:
        b.append("moderate_timing_noise")
    if dropped > cfg["a_max_dropped_frame_fraction"]:
        b.append("some_dropped_frames")
    if not session.get("fullscreen", False):
        b.append("not_fullscreen")
    if failed == 1:
        b.append("one_failed_attention_check")
    return ("B", b) if b else ("A", [])
