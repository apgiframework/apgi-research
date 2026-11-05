"""Backward-masking timing and output helpers (S3-BAT-01.3.1-.5).

Analysis-side only: the browser task itself is built in apgiframework.com and is
gated on decision S3-DEC-01 (masking vs attentional blink). These helpers are
decision-agnostic - frame locking, achieved-SOA logging and PAS handling do not
depend on which paradigm the decision selects."""

import math
import statistics
from collections import defaultdict
from typing import Any, Optional, Sequence

PAS_LEVELS = (1, 2, 3, 4)  # 1 no experience .. 4 clear experience (Ramsøy & Overgaard, 2004)


def lock_soa_to_frames(soa_ms: float, refresh_ms: float) -> dict[str, float]:
    """S3-BAT-01.3.1: nearest whole number of frames (>= 1)."""
    if refresh_ms <= 0 or soa_ms <= 0:
        raise ValueError("soa_ms and refresh_ms must be positive")
    n = max(1, round(soa_ms / refresh_ms))
    return {"frames": n, "locked_soa_ms": n * refresh_ms, "requested_soa_ms": soa_ms,
            "rounding_error_ms": n * refresh_ms - soa_ms}


def achieved_soa(target_onset_ms: float, mask_onset_ms: float, requested_frames: int,
                 refresh_ms: float) -> dict[str, Any]:
    """S3-BAT-01.3.2: the per-trial log line. A frame error of >= 1 frame marks
    the trial as timing-failed."""
    achieved = mask_onset_ms - target_onset_ms
    frames = achieved / refresh_ms
    return {"requested_frames": requested_frames, "achieved_soa_ms": achieved,
            "achieved_frames": frames, "frame_error": round(frames) - requested_frames,
            "dropped": abs(round(frames) - requested_frames) >= 1}


def validate_pas(rating: int) -> int:
    """S3-BAT-01.3.3: per-trial 4-point Perceptual Awareness Scale."""
    if rating not in PAS_LEVELS:
        raise ValueError(f"PAS rating must be one of {PAS_LEVELS}")
    return rating


def downtier_for_dropped_frames(trials: Sequence[dict[str, Any]], tier: str,
                                max_fraction: float = 0.05) -> tuple[str, list[str]]:
    """S3-BAT-01.3.4: a session whose share of dropped-SOA trials exceeds the
    limit moves down one tier (A->B, B->C); C stays C."""
    if not trials:
        return "C", ["no_trials"]
    frac = sum(bool(t["dropped"]) for t in trials) / len(trials)
    if frac <= max_fraction:
        return tier, []
    return {"A": "B", "B": "C", "C": "C"}[tier], [f"dropped_soa_fraction={frac:.3f}"]


def _acc(rows: Sequence[dict[str, Any]]) -> float:
    return sum(bool(r["correct"]) for r in rows) / len(rows)


def summarize(trials: Sequence[dict[str, Any]], half_split_seed: Optional[int] = None) -> dict[str, Any]:
    """S3-BAT-01.3.5: accuracy by SOA, RT distribution, confidence, drift and
    split-half reliability. Trials need soa_frames, correct, rt_ms, confidence,
    trial_index."""
    if not trials:
        raise ValueError("no trials")
    by_soa: dict[int, list] = defaultdict(list)
    for t in trials:
        by_soa[int(t["soa_frames"])].append(t)
    acc = {k: {"accuracy": _acc(v), "n": len(v),
               "mean_confidence": statistics.fmean(x["confidence"] for x in v)}
           for k, v in sorted(by_soa.items())}
    rts = sorted(float(t["rt_ms"]) for t in trials)

    def q(p: float) -> float:
        return rts[min(len(rts) - 1, int(round(p * (len(rts) - 1))))]

    ordered = sorted(trials, key=lambda t: t["trial_index"])
    half = len(ordered) // 2
    drift = (_acc(ordered[half:]) - _acc(ordered[:half])) if half else float("nan")
    # Split halves alternate *within* each SOA so a design in which SOA is tied
    # to trial-index parity cannot empty a half.
    odd: list[dict[str, Any]] = []
    even: list[dict[str, Any]] = []
    seen: dict[int, int] = defaultdict(int)
    for t in ordered:
        k = int(t["soa_frames"])
        (odd if seen[k] % 2 else even).append(t)
        seen[k] += 1
    rel = float("nan")
    if len(by_soa) >= 3:
        ks = sorted(by_soa)
        a = [_acc([t for t in odd if int(t["soa_frames"]) == k]) for k in ks]
        b = [_acc([t for t in even if int(t["soa_frames"]) == k]) for k in ks]
        if statistics.pstdev(a) > 0 and statistics.pstdev(b) > 0:
            r = statistics.correlation(a, b)
            rel = 2 * r / (1 + r) if r > -1 else float("nan")  # Spearman-Brown
    return {"accuracy_by_soa": acc,
            "rt_ms": {"median": q(0.5), "p05": q(0.05), "p95": q(0.95), "n": len(rts)},
            "accuracy_drift_second_minus_first_half": drift,
            "split_half_reliability_within_soa": rel,
            "reliability_note": "profile-level; trial-level reliability needs pilot data"}
