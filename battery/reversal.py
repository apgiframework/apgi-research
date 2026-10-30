"""Reversal-learning model fit (S3-BAT-01.2.2) and the only reported composite
(S3-BAT-01.2.3)."""

from typing import Sequence

import numpy as np
from scipy.optimize import minimize


def _nll(params: np.ndarray, choices: np.ndarray, rewards: np.ndarray) -> float:
    a_pos, a_neg, beta, kappa = params
    q = np.zeros(2)
    prev = -1
    total = 0.0
    for choice, reward in zip(choices, rewards):
        logits = beta * q
        if prev >= 0:
            logits[prev] += kappa  # perseveration bonus for repeating the last choice
        logits -= logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        total -= np.log(max(p[choice], 1e-12))
        delta = reward - q[choice]
        q[choice] += (a_pos if delta >= 0 else a_neg) * delta
        prev = choice
    return float(total)


def fit_reversal(
    choices: Sequence[int], rewards: Sequence[float], seed: int = 0, restarts: int = 6
) -> dict[str, float]:
    """Q-learning with separate learning rates for positive and negative
    prediction errors, inverse temperature (reward sensitivity) and
    perseveration. Multi-start L-BFGS-B, deterministic for a given seed."""
    ch = np.asarray(choices, dtype=int)
    rw = np.asarray(rewards, dtype=float)
    if len(ch) != len(rw) or len(ch) < 20:
        raise ValueError("need >= 20 trials with matching choices and rewards")
    if not set(np.unique(ch)) <= {0, 1}:
        raise ValueError("choices must be 0 or 1")
    rng = np.random.default_rng(seed)
    bounds = [(0.01, 1.0), (0.01, 1.0), (0.1, 20.0), (-3.0, 3.0)]
    best = None
    for _ in range(restarts):
        x0 = [
            rng.uniform(0.1, 0.9),
            rng.uniform(0.1, 0.9),
            rng.uniform(1, 6),
            rng.uniform(-0.5, 0.5),
        ]
        res = minimize(_nll, x0, args=(ch, rw), method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    a_pos, a_neg, beta, kappa = (float(v) for v in best.x)
    return {
        "alpha_positive": a_pos,
        "alpha_negative": a_neg,
        "reward_sensitivity": beta,
        "perseveration": kappa,
        "neg_log_likelihood": float(best.fun),
    }


def precision_weighted_learning_composite(
    alpha: float, precision: float
) -> dict[str, float]:
    """The single reported learning index: the alpha x Pi product. Returns only
    this key so component values cannot leak into reports."""
    if alpha < 0 or precision < 0:
        raise ValueError("alpha and precision must be non-negative")
    return {"precision_weighted_learning_composite": alpha * precision}
