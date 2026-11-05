"""Display and viewing-distance calibration, frame measurement and attention
checks (S3-BAT-01.4.1-.4)."""

import math
import statistics
from typing import Any, Optional, Sequence

CREDIT_CARD_WIDTH_MM = 85.60  # ISO/IEC 7810 ID-1
BLIND_SPOT_ECCENTRICITY_DEG = 13.5  # Li et al., 2020 (Virtual Chinrest)


def measure_display_timing(frame_timestamps_ms: Sequence[float]) -> dict[str, Any]:
    """S3-BAT-01.4.1: FPS, frame-interval distribution and dropped frames from
    consecutive rAF timestamps (performance.now(), ms)."""
    ts = [float(t) for t in frame_timestamps_ms]
    if len(ts) < 10 or not all(math.isfinite(t) for t in ts):
        raise ValueError("need >= 10 finite frame timestamps")
    dts = [b - a for a, b in zip(ts, ts[1:])]
    if any(d <= 0 for d in dts):
        raise ValueError("frame timestamps must be strictly increasing")
    med = statistics.median(dts)
    refresh_hz = 1000.0 / med
    dropped = sum(d > 1.5 * med for d in dts)
    xs = sorted(abs(d - med) for d in dts)
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    return {
        "refresh_estimate_ms": med,
        "fps": refresh_hz,
        "n_intervals": len(dts),
        "dropped_frame_fraction": dropped / len(dts),
        "interval_sd_ms": statistics.pstdev(dts),
        "interval_p95_abs_dev_ms": p95,
    }


def pixels_per_mm(card_width_px: float, card_width_mm: float = CREDIT_CARD_WIDTH_MM) -> float:
    """S3-BAT-01.4.2: card-against-screen scale. The participant resizes an
    on-screen bar to the card's long edge."""
    if not (math.isfinite(card_width_px) and card_width_px > 0):
        raise ValueError("card_width_px must be positive and finite")
    return card_width_px / card_width_mm


def viewing_distance_mm(
    blind_spot_offset_px: Sequence[float],
    px_per_mm: float,
    eccentricity_deg: float = BLIND_SPOT_ECCENTRICITY_DEG,
) -> dict[str, float]:
    """S3-BAT-01.4.3: Virtual-Chinrest estimate (Li et al., 2020). The blind
    spot lies ~13.5 deg from fixation; with the measured fixation-to-blind-spot
    distance d (mm on screen) the viewing distance is d / tan(13.5 deg).
    Several repeats are averaged; their spread is returned as uncertainty."""
    xs = [float(x) for x in blind_spot_offset_px]
    if len(xs) < 3 or not all(math.isfinite(x) and x > 0 for x in xs):
        raise ValueError("need >= 3 positive finite blind-spot measurements")
    if px_per_mm <= 0:
        raise ValueError("px_per_mm must be positive")
    k = math.tan(math.radians(eccentricity_deg))
    dists = [x / px_per_mm / k for x in xs]
    return {
        "viewing_distance_mm": statistics.mean(dists),
        "sd_mm": statistics.stdev(dists),
        "n": len(dists),
        "method": "blind_spot_li_2020",
    }


def pixels_per_degree(px_per_mm: float, distance_mm: float) -> float:
    return px_per_mm * distance_mm * math.tan(math.radians(1.0))


def score_attention_checks(
    expected: Sequence[Any], given: Sequence[Any], max_failed: int = 2
) -> dict[str, Any]:
    """S3-BAT-01.4.4: attention-check outcome; >= `max_failed` fails excludes
    from norms (shares the threshold with screening)."""
    if len(expected) != len(given):
        raise ValueError("expected and given differ in length")
    failed = sum(e != g for e, g in zip(expected, given))
    return {"n_checks": len(expected), "failed": failed, "excluded_from_norms": failed >= max_failed}
