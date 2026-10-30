"""Signal-detection indices (S3-BAT-01.1.3): d' = z(H) - z(FA), c = -1/2 [z(H) + z(FA)]."""

from statistics import NormalDist
from typing import Optional, Sequence

_Z = NormalDist().inv_cdf


def rates(
    hits: int, misses: int, false_alarms: int, correct_rejections: int
) -> tuple[float, float]:
    """Hit and false-alarm rates with the log-linear correction (add 0.5 to each
    cell), so extreme counts never give infinite z-scores."""
    h = (hits + 0.5) / (hits + misses + 1.0)
    f = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1.0)
    return h, f


def dprime_c(
    hits: int, misses: int, false_alarms: int, correct_rejections: int
) -> dict[str, float]:
    if min(hits, misses, false_alarms, correct_rejections) < 0:
        raise ValueError("counts must be non-negative")
    if hits + misses == 0 or false_alarms + correct_rejections == 0:
        raise ValueError("need at least one signal-present and one signal-absent trial")
    h, f = rates(hits, misses, false_alarms, correct_rejections)
    return {
        "d_prime": _Z(h) - _Z(f),
        "criterion_c": -0.5 * (_Z(h) + _Z(f)),
        "hit_rate": h,
        "fa_rate": f,
    }


def counts_from_trials(
    present: Sequence[bool], said_present: Sequence[bool]
) -> tuple[int, int, int, int]:
    if len(present) != len(said_present):
        raise ValueError("present and said_present differ in length")
    hits = sum(p and s for p, s in zip(present, said_present))
    misses = sum(p and not s for p, s in zip(present, said_present))
    fas = sum((not p) and s for p, s in zip(present, said_present))
    crs = sum((not p) and (not s) for p, s in zip(present, said_present))
    return hits, misses, fas, crs


def dprime_by_presence(
    present: Sequence[bool],
    said_present: Sequence[bool],
    confidence: Optional[Sequence[float]] = None,
) -> dict[str, float]:
    """d', c from trial vectors; adds mean confidence for present/absent trials
    when given (inputs to the separate present/absent meta-d' analysis)."""
    out = dprime_c(*counts_from_trials(present, said_present))
    if confidence is not None:
        for label, flag in (("present", True), ("absent", False)):
            vals = [c for p, c in zip(present, confidence) if p == flag]
            out[f"mean_confidence_{label}"] = (
                sum(vals) / len(vals) if vals else float("nan")
            )
    return out
