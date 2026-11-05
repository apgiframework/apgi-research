"""Individual contrast calibration (S3-BAT-01.1.1): a 3-down/1-up transformed
staircase (Levitt, 1971; converges near 79.4% correct) and a QUEST-style
Bayesian posterior over log10 threshold (Watson & Pelli, 1983)."""

import math
from statistics import NormalDist
from typing import Optional

import numpy as np


class Staircase:
    """3-down/1-up on log10 contrast. Deterministic: feed it outcomes."""

    def __init__(self, start_log10: float = -1.0, step_log10: float = 0.1,
                 lo: float = -3.0, hi: float = 0.0, n_down: int = 3,
                 min_reversals: int = 8) -> None:
        if not lo < start_log10 <= hi:
            raise ValueError("start must lie within (lo, hi]")
        self.level, self.step, self.lo, self.hi = start_log10, step_log10, lo, hi
        self.n_down, self.min_reversals = n_down, min_reversals
        self._streak = 0
        self._direction = 0
        self.reversals: list[float] = []
        self.history: list[tuple[float, bool]] = []

    @property
    def contrast(self) -> float:
        return 10 ** self.level

    def update(self, correct: bool) -> float:
        self.history.append((self.level, bool(correct)))
        move = 0
        if correct:
            self._streak += 1
            if self._streak >= self.n_down:
                self._streak, move = 0, -1
        else:
            self._streak, move = 0, +1
        if move:
            if self._direction and move != self._direction:
                self.reversals.append(self.level)
            self._direction = move
            self.level = min(self.hi, max(self.lo, self.level + move * self.step))
        return self.contrast

    @property
    def done(self) -> bool:
        return len(self.reversals) >= self.min_reversals

    def threshold(self, drop: int = 2) -> Optional[float]:
        """Mean of reversals after discarding the first `drop`; contrast units."""
        use = self.reversals[drop:]
        if len(use) < 4:
            return None
        return 10 ** (sum(use) / len(use))


class Quest:
    """Posterior over log10 threshold with a Weibull psychometric function."""

    def __init__(self, prior_mean: float = -1.5, prior_sd: float = 0.6,
                 beta: float = 3.5, gamma: float = 0.5, delta: float = 0.02,
                 grid: Optional[np.ndarray] = None) -> None:
        self.grid = np.linspace(-3.0, 0.0, 601) if grid is None else np.asarray(grid)
        self.beta, self.gamma, self.delta = beta, gamma, delta
        nd = NormalDist(prior_mean, prior_sd)
        self.log_post = np.array([math.log(max(nd.pdf(g), 1e-300)) for g in self.grid])

    def _p_correct(self, x: float) -> np.ndarray:
        core = 1.0 - np.exp(-(10 ** (self.beta * (x - self.grid))))
        return self.gamma + (1 - self.gamma - self.delta) * core

    def update(self, log10_contrast: float, correct: bool) -> None:
        p = np.clip(self._p_correct(log10_contrast), 1e-9, 1 - 1e-9)
        self.log_post += np.log(p if correct else 1 - p)
        self.log_post -= self.log_post.max()

    def next_level(self) -> float:
        """Place the next trial at the posterior mean (Quest default)."""
        w = np.exp(self.log_post)
        return float((w * self.grid).sum() / w.sum())

    def estimate(self) -> dict[str, float]:
        w = np.exp(self.log_post)
        w /= w.sum()
        mean = float((w * self.grid).sum())
        sd = float(math.sqrt((w * (self.grid - mean) ** 2).sum()))
        return {"log10_threshold": mean, "threshold": 10 ** mean, "posterior_sd": sd}
