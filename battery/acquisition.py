"""Acquisition integrity checks (APGI-ACQ-001): record sampling parameters and a
checksum with every recording and detect drift, dropped samples and corruption."""

import hashlib
import math
from typing import Any, Optional, Sequence


def recording_header(samples: Sequence[float], timestamps_ms: Sequence[float], nominal_hz: float) -> dict[str, Any]:
    if nominal_hz <= 0 or len(samples) != len(timestamps_ms) or len(samples) < 2:
        raise ValueError("need >= 2 matching samples/timestamps and positive nominal_hz")
    digest = hashlib.sha256(
        b"".join(f"{s!r}|{t!r};".encode() for s, t in zip(samples, timestamps_ms))
    ).hexdigest()
    return {"n_samples": len(samples), "nominal_hz": nominal_hz,
            "first_timestamp_ms": timestamps_ms[0], "last_timestamp_ms": timestamps_ms[-1],
            "sha256": digest}


def check_recording(
    samples: Sequence[float], timestamps_ms: Sequence[float], header: dict[str, Any],
    max_drift_ppm: float = 100.0, max_dropped_fraction: float = 0.0,
) -> dict[str, Any]:
    """Verify checksum, count dropped samples (gaps > 1.5 nominal intervals) and
    clock drift against the nominal rate. `max_*` are placeholders for the
    protocol-approved tolerances; they must be set from the registered acquisition
    contract before use in a study."""
    problems: list[str] = []
    fresh = recording_header(samples, timestamps_ms, header["nominal_hz"])
    if fresh["sha256"] != header["sha256"] or fresh["n_samples"] != header["n_samples"]:
        problems.append("checksum_or_length_mismatch")
    if any(not math.isfinite(x) for x in samples):
        problems.append("nonfinite_samples")
    dt = 1000.0 / header["nominal_hz"]
    gaps = [b - a for a, b in zip(timestamps_ms, timestamps_ms[1:])]
    if any(g <= 0 for g in gaps):
        problems.append("non_monotonic_timestamps")
    dropped = sum(max(0, round(g / dt) - 1) for g in gaps if g > 0 and g > 1.5 * dt)
    expected = (timestamps_ms[-1] - timestamps_ms[0]) / dt + 1
    drift_ppm = ((len(samples) - 1) * dt - (timestamps_ms[-1] - timestamps_ms[0])) / (
        (len(samples) - 1) * dt) * 1e6 if len(samples) > 1 else 0.0
    frac = dropped / expected if expected else 0.0
    if frac > max_dropped_fraction:
        problems.append("dropped_samples")
    if abs(drift_ppm) > max_drift_ppm and not dropped:
        problems.append("clock_drift")
    return {"ok": not problems, "problems": problems, "dropped_samples": dropped,
            "dropped_fraction": frac, "drift_ppm": drift_ppm}
