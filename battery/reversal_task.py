"""Reversal-learning task schedule (S3-BAT-01.2.1): two options, probabilistic
rewards, unannounced reversals. The schedule is fully determined by the seed
and recorded with the session so a run is reproducible (APGI-REP-002)."""

import numpy as np


def make_schedule(seed: int, n_trials: int = 120, p_good: float = 0.8,
                  run_min: int = 12, run_max: int = 24) -> dict:
    if not 0.5 < p_good < 1.0:
        raise ValueError("p_good must be in (0.5, 1)")
    if not 1 <= run_min <= run_max < n_trials:
        raise ValueError("invalid run lengths")
    rng = np.random.default_rng(seed)
    good = np.zeros(n_trials, dtype=int)
    t, opt, reversals = 0, int(rng.integers(2)), []
    while t < n_trials:
        run = int(rng.integers(run_min, run_max + 1))
        good[t:t + run] = opt
        t += run
        if t < n_trials:
            reversals.append(t)
        opt = 1 - opt
    draws = rng.random((n_trials, 2))
    reward = np.zeros((n_trials, 2), dtype=int)
    for o in (0, 1):
        p = np.where(good == o, p_good, 1 - p_good)
        reward[:, o] = (draws[:, o] < p).astype(int)
    return {"seed": seed, "n_trials": n_trials, "p_good": p_good,
            "good_option": good.tolist(), "reward": reward.tolist(),
            "reversal_trials": reversals, "announced": False}
