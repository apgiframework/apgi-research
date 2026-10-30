"""Psychometric functions for performance and confidence (S3-BAT-01.1.4) and
device-relative contrast thresholds (S3-BAT-01.8)."""

from typing import Any, Optional, Sequence

import numpy as np
from scipy.optimize import minimize


def _logistic(
    x: np.ndarray, mu: float, sigma: float, gamma: float, lam: float
) -> np.ndarray:
    core = 1.0 / (1.0 + np.exp(-(x - mu) / sigma))
    return gamma + (1.0 - gamma - lam) * core


def fit_psychometric(
    contrast: Sequence[float],
    n_correct: Sequence[int],
    n_trials: Sequence[int],
    gamma: float = 0.0,
    lapse: float = 0.02,
) -> dict[str, float]:
    """Maximum-likelihood logistic fit on log10 contrast.

    `gamma` is the guess rate (0 for yes/no detection with a separate false-alarm
    model, 0.5 for 2AFC). Returns the 50%-point (`threshold`, same units as
    `contrast`) and `slope` (1/sigma in log10 units).
    """
    c = np.asarray(contrast, dtype=float)
    k = np.asarray(n_correct, dtype=float)
    n = np.asarray(n_trials, dtype=float)
    if np.any(c <= 0):
        raise ValueError("contrast must be positive")
    if np.any(k > n) or np.any(n <= 0):
        raise ValueError("invalid trial counts")
    x = np.log10(c)

    def nll(theta: np.ndarray) -> float:
        mu, log_sigma = theta
        p = np.clip(
            _logistic(x, mu, float(np.exp(log_sigma)), gamma, lapse), 1e-9, 1 - 1e-9
        )
        return float(-(k * np.log(p) + (n - k) * np.log(1 - p)).sum())

    best = minimize(nll, x0=[float(np.median(x)), np.log(0.2)], method="Nelder-Mead")
    mu, sigma = float(best.x[0]), float(np.exp(best.x[1]))
    return {
        "threshold": 10**mu,
        "log10_threshold": mu,
        "slope": 1.0 / sigma,
        "neg_log_likelihood": float(best.fun),
    }


def fit_confidence_function(
    contrast: Sequence[float], mean_confidence: Sequence[float]
) -> dict[str, float]:
    """Logistic fit of mean confidence (rescaled 0-1) against log10 contrast."""
    c = np.log10(np.asarray(contrast, dtype=float))
    y = np.asarray(mean_confidence, dtype=float)
    lo, hi = float(y.min()), float(y.max())
    if hi == lo:
        raise ValueError("confidence does not vary with contrast")
    y01 = (y - lo) / (hi - lo)

    def sse(theta: np.ndarray) -> float:
        mu, log_sigma = theta
        return float(
            ((_logistic(c, mu, float(np.exp(log_sigma)), 0.0, 0.0) - y01) ** 2).sum()
        )

    best = minimize(sse, x0=[float(np.median(c)), np.log(0.2)], method="Nelder-Mead")
    return {
        "midpoint": 10 ** float(best.x[0]),
        "slope": 1.0 / float(np.exp(best.x[1])),
        "sse": float(best.fun),
    }


def device_relative_threshold(
    threshold_contrast: float,
    device: dict[str, Any],
    tier: str,
    tier_config_version: Optional[str] = None,
) -> dict[str, Any]:
    """Report a contrast threshold as device-relative, never as an absolute
    population value, with tier and display covariates attached."""
    required = ("refresh_hz", "gamma", "bit_depth", "screen_px")
    missing = [k for k in required if k not in device]
    if missing:
        raise ValueError(f"device covariates missing: {missing}")
    if tier not in ("A", "B", "C"):
        raise ValueError("tier must be A, B or C")
    lsb = 1.0 / (2 ** int(device["bit_depth"]) - 1)
    return {
        "contrast_threshold": float(threshold_contrast),
        "threshold_in_display_steps": float(threshold_contrast) / lsb,
        "scale": "device_relative",
        "comparable_across_devices": False,
        "covariates": {
            "tier": tier,
            "tier_config_version": tier_config_version,
            **{k: device[k] for k in required},
        },
    }
